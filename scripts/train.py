#!/usr/bin/env python3
"""Post-training with GRPO (Group Relative Policy Optimization).

This is the central experiment: update model weights using reward signal
from our compositional reward function.

Usage:
    PYTHONPATH=. python3.11 scripts/train.py
    PYTHONPATH=. python3.11 scripts/train.py --method dpo  # fallback
    PYTHONPATH=. python3.11 scripts/train.py --reward-weights configs/reward_weights.yaml
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer, DPOConfig, DPOTrainer

# Compatibility fix: transformers 4.57 calls _get_train_sampler(dataset)
# but TRL 0.15.2 defines it as _get_train_sampler(self) with no args.
# Patch to accept and ignore the extra argument.
_orig_sampler = GRPOTrainer._get_train_sampler
def _patched_sampler(self, train_dataset=None):
    return _orig_sampler(self)
GRPOTrainer._get_train_sampler = _patched_sampler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.environment.state import Scenario
from src.policies.prompt_templates import (
    SYSTEM_PROMPT_ENGINEERED,
    build_messages,
)
from src.policies.inference import extract_json_from_text
from src.rewards.reward import RewardCalculator
from src.environment.actions import parse_model_output


SEED = 42
torch.manual_seed(SEED)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "qwen2.5-1.5b-instruct"
)
TRAIN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "train.json"
)
VAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "val.json"
)
OUTPUT_DIR_GRPO = os.path.join(
    os.path.dirname(__file__), "..", "models", "trained_rl"
)
OUTPUT_DIR_DPO = os.path.join(
    os.path.dirname(__file__), "..", "models", "trained_dpo"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results"
)


def load_scenarios(path):
    with open(path) as f:
        data = json.load(f)
    return [Scenario.from_dict(d) for d in data]


def build_prompt_text(scenario, tokenizer):
    """Build the full prompt text for a scenario."""
    messages = build_messages(
        scenario.state, SYSTEM_PROMPT_ENGINEERED
    )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# GRPO reward function — closure over scenario ground truth
# ---------------------------------------------------------------------------

def make_grpo_reward_fn(scenarios: List[Scenario], reward_calc: RewardCalculator):
    """Create a GRPO reward function that uses ground truth.

    TRL 0.15.2 GRPOTrainer calls reward functions as:
        reward_func(prompts=prompts, completions=completions, **reward_kwargs)
    where reward_kwargs includes any extra columns from the dataset
    (here: scenario_id).

    Returns a closure that maps scenario_id -> Scenario to compute
    real composite rewards via RewardCalculator.
    """
    scenario_map: Dict[str, Scenario] = {
        s.scenario_id: s for s in scenarios
    }

    _log_count = {"n": 0}  # mutable counter for logging first N completions

    def reward_fn(prompts, completions, scenario_id, **kwargs):
        rewards = []
        for completion, sid in zip(completions, scenario_id):
            # Extract text from completion
            text = completion
            if isinstance(completion, list):
                # Chat format: list of message dicts
                text = completion[-1]["content"] if completion else ""

            text = str(text)

            # Parse JSON from model output
            parsed = extract_json_from_text(text)
            if parsed is None:
                rewards.append(0.0)
                if _log_count["n"] < 5:
                    _log_count["n"] += 1
                    print(f"  [reward] sid={sid} -> 0.0 (unparseable JSON)")
                continue

            # Parse into ModelOutput
            output = parse_model_output(parsed)
            if output is None or not output.is_valid:
                rewards.append(0.0)
                if _log_count["n"] < 5:
                    _log_count["n"] += 1
                    print(f"  [reward] sid={sid} -> 0.0 (invalid ModelOutput)")
                continue

            # Look up ground truth
            scenario = scenario_map.get(sid)
            if scenario is None:
                rewards.append(0.0)
                if _log_count["n"] < 5:
                    _log_count["n"] += 1
                    print(f"  [reward] sid={sid} -> 0.0 (scenario not found)")
                continue

            # Compute real composite reward against ground truth
            total_reward, breakdown = reward_calc.compute(
                output, scenario.ground_truth
            )
            rewards.append(float(total_reward))

            # Log first few completions with full breakdown
            if _log_count["n"] < 5:
                _log_count["n"] += 1
                components_str = ", ".join(
                    f"{k}={v:.2f}"
                    for k, v in breakdown["components"].items()
                )
                print(
                    f"  [reward] sid={sid} -> {total_reward:.4f} "
                    f"(action={output.action}, intent={output.intent_assessment}) "
                    f"[{components_str}]"
                )

        return rewards

    return reward_fn


def prepare_grpo_dataset(scenarios, tokenizer):
    """Prepare dataset for GRPO training.

    Includes scenario_id column so the trainer passes it to the
    reward function as a keyword argument.
    """
    prompts = []
    scenario_ids = []
    for scenario in scenarios:
        prompt = build_prompt_text(scenario, tokenizer)
        prompts.append(prompt)
        scenario_ids.append(scenario.scenario_id)

    return Dataset.from_dict({
        "prompt": prompts,
        "scenario_id": scenario_ids,
    })


def prepare_dpo_dataset(
    scenarios, model, tokenizer, reward_calc, device, n_gen=4
):
    """Generate preference pairs for DPO training.

    For each scenario:
    1. Generate n_gen completions
    2. Score each with reward function
    3. Pick best as chosen, worst as rejected
    """
    from src.policies.inference import PolicyRunner

    runner = PolicyRunner(
        model=model,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT_ENGINEERED,
        max_new_tokens=512,
        temperature=0.8,
        device=device,
    )

    chosen_list = []
    rejected_list = []
    prompt_list = []
    skipped = 0

    for i, scenario in enumerate(scenarios):
        print(f"  DPO pair gen: {i+1}/{len(scenarios)}", end="\r")
        try:
            responses = runner.generate(
                scenario.state, num_return=n_gen
            )

            # Score each response
            scored = []
            for resp in responses:
                parsed = extract_json_from_text(resp)
                if parsed:
                    output = parse_model_output(parsed)
                    if output and output.is_valid:
                        r, _ = reward_calc.compute(
                            output, scenario.ground_truth
                        )
                        scored.append((r, resp))

            if len(scored) < 2:
                skipped += 1
                continue

            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            worst = scored[-1][1]

            # Only keep if there's a meaningful reward gap
            if scored[0][0] - scored[-1][0] >= 0.1:
                prompt = build_prompt_text(scenario, tokenizer)
                prompt_list.append(prompt)
                chosen_list.append(best)
                rejected_list.append(worst)

        except Exception as e:
            skipped += 1
            continue

    print(f"\n  Generated {len(prompt_list)} preference pairs, "
          f"skipped {skipped}")

    return Dataset.from_dict({
        "prompt": prompt_list,
        "chosen": chosen_list,
        "rejected": rejected_list,
    })


def train_grpo(model, tokenizer, train_dataset, grpo_reward_fn, device):
    """Train with GRPO."""
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    # MPS on PyTorch < 2.5 doesn't support fp16 mixed precision
    # via Accelerate. Use fp32 on MPS, fp16 on CUDA.
    use_fp16 = device == "cuda"

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR_GRPO,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        max_completion_length=256,
        max_prompt_length=512,
        num_generations=4,
        beta=0.1,
        temperature=0.7,
        seed=SEED,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        bf16=False,
        fp16=use_fp16,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting GRPO training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR_GRPO)
    tokenizer.save_pretrained(OUTPUT_DIR_GRPO)
    print(f"Model saved to {OUTPUT_DIR_GRPO}")
    return trainer


def train_dpo(model, tokenizer, train_dataset, device):
    """Train with DPO (fallback)."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    use_fp16 = device == "cuda"

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR_DPO,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_length=1280,
        max_prompt_length=768,
        beta=0.1,
        seed=SEED,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        bf16=False,
        fp16=use_fp16,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR_DPO)
    tokenizer.save_pretrained(OUTPUT_DIR_DPO)
    print(f"Model saved to {OUTPUT_DIR_DPO}")
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["grpo", "dpo"], default="grpo",
        help="Training method (default: grpo)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run on 20 examples only"
    )
    parser.add_argument(
        "--reward-weights", type=str, default=None,
        help="Path to YAML file with reward weights for ablation studies"
    )
    args = parser.parse_args()

    output_dir = OUTPUT_DIR_GRPO if args.method == "grpo" else OUTPUT_DIR_DPO
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load data
    train_scenarios = load_scenarios(TRAIN_PATH)
    if args.smoke_test:
        train_scenarios = train_scenarios[:20]
    print(f"Training on {len(train_scenarios)} scenarios")

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in fp16 on all devices. The MPS fp16 Accelerate issue is
    # specifically about Accelerate's mixed precision mode (fp16=True in config),
    # not about the model's native dtype. We keep fp16=False in GRPOConfig
    # but load the model weights as fp16 for speed.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print("Model loaded.")

    # Build reward calculator (optionally from YAML)
    if args.reward_weights:
        print(f"Loading reward weights from {args.reward_weights}")
        reward_calc = RewardCalculator.from_yaml(args.reward_weights)
    else:
        reward_calc = RewardCalculator()

    t0 = time.time()

    if args.method == "grpo":
        print("\n=== GRPO Training ===")
        dataset = prepare_grpo_dataset(
            train_scenarios, tokenizer
        )
        grpo_reward_fn = make_grpo_reward_fn(
            train_scenarios, reward_calc
        )
        print(f"Dataset size: {len(dataset)} prompts")
        print(f"Reward weights: {reward_calc.weights}")
        trainer = train_grpo(
            model, tokenizer, dataset, grpo_reward_fn, device
        )
    else:
        print("\n=== DPO Training (preference optimization) ===")
        print("Generating preference pairs from reward function...")
        # Need model on device for generation
        model = model.to(device)
        dataset = prepare_dpo_dataset(
            train_scenarios, model, tokenizer,
            reward_calc, device, n_gen=4
        )
        # Move model back to CPU for DPO trainer to handle
        model = model.to("cpu")
        trainer = train_dpo(model, tokenizer, dataset, device)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Save training metadata
    meta = {
        "method": args.method,
        "n_train_scenarios": len(train_scenarios),
        "training_time_minutes": round(elapsed / 60, 1),
        "device": device,
        "model": MODEL_PATH,
        "seed": SEED,
        "smoke_test": args.smoke_test,
        "reward_weights": reward_calc.weights,
        "reward_weights_source": args.reward_weights or "default",
    }
    with open(
        os.path.join(RESULTS_DIR, "training_meta.json"), "w"
    ) as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
