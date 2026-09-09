#!/usr/bin/env python3
"""Unified evaluation script for any model condition on any dataset split.

Usage:
    # Base model with engineered prompt on test set
    PYTHONPATH=. python3.11 scripts/evaluate.py --condition engineered --split test

    # RL-trained model on test set
    PYTHONPATH=. python3.11 scripts/evaluate.py --condition rl --split test

    # DPO-trained model on challenge set
    PYTHONPATH=. python3.11 scripts/evaluate.py --condition dpo --split challenge

    # Custom reward weights (ablation)
    PYTHONPATH=. python3.11 scripts/evaluate.py --condition rl --split test \
        --reward-weights configs/reward_weights_no_efficiency.yaml
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.environment.state import Scenario
from src.environment.actions import (
    VALID_ACTIONS,
    VALID_INTENTS,
    parse_model_output,
)
from src.policies.prompt_templates import (
    SYSTEM_PROMPT_MINIMAL,
    SYSTEM_PROMPT_ENGINEERED,
    build_messages,
)
from src.policies.inference import PolicyRunner
from src.rewards.reward import RewardCalculator
from src.evaluation.metrics import (
    compute_metrics,
    print_confusion_matrix,
    bootstrap_ci,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b-instruct")

VALID_CONDITIONS = ("random", "zero_shot", "engineered", "dpo", "rl")
VALID_SPLITS = ("test", "val", "challenge")

CONDITION_CONFIG = {
    "random": {
        "needs_model": False,
        "adapter_path": None,
        "system_prompt": None,
        "description": "Random baseline (no model)",
    },
    "zero_shot": {
        "needs_model": True,
        "adapter_path": None,
        "system_prompt": SYSTEM_PROMPT_MINIMAL,
        "description": "Base model + minimal prompt",
    },
    "engineered": {
        "needs_model": True,
        "adapter_path": None,
        "system_prompt": SYSTEM_PROMPT_ENGINEERED,
        "description": "Base model + engineered prompt",
    },
    "dpo": {
        "needs_model": True,
        "adapter_path": os.path.join(
            PROJECT_ROOT, "models", "trained_dpo"
        ),
        "system_prompt": SYSTEM_PROMPT_ENGINEERED,
        "description": "DPO-trained model + engineered prompt",
    },
    "rl": {
        "needs_model": True,
        "adapter_path": os.path.join(
            PROJECT_ROOT, "models", "trained_rl"
        ),
        "system_prompt": SYSTEM_PROMPT_ENGINEERED,
        "description": "RL-trained model + engineered prompt",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model condition on a dataset split."
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=VALID_CONDITIONS,
        help="Evaluation condition: random, zero_shot, engineered, dpo, rl",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=VALID_SPLITS,
        help="Dataset split: test, val, challenge",
    )
    parser.add_argument(
        "--reward-weights",
        default=None,
        help="Path to reward weights YAML (optional, for ablation)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Inference temperature (default: 0.3)",
    )
    return parser.parse_args()


def get_device() -> str:
    """Determine best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_scenarios(split: str):
    """Load scenarios from the given split file."""
    if split == "challenge":
        data_path = os.path.join(
            PROJECT_ROOT, "data", "challenge", "scenarios.json"
        )
    else:
        data_path = os.path.join(
            PROJECT_ROOT, "data", "processed", f"{split}.json"
        )
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)
    return [Scenario.from_dict(d) for d in data]


def load_model_and_tokenizer(condition: str, device: str):
    """Load model (+ optional LoRA adapter) and tokenizer.

    Returns (model, tokenizer).
    """
    config = CONDITION_CONFIG[condition]

    print(f"Loading base model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    # Apply LoRA adapter if needed
    adapter_path = config["adapter_path"]
    if adapter_path is not None:
        if not os.path.exists(adapter_path):
            print(
                f"ERROR: Adapter path does not exist: {adapter_path}"
            )
            sys.exit(1)

        print(f"Loading LoRA adapter from {adapter_path} ...")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged and unloaded.")

    model.eval()
    print("Model ready.")
    return model, tokenizer


def run_random_baseline(scenarios):
    """Random baseline: pick random action/intent for each scenario."""
    intents = list(VALID_INTENTS)
    actions = list(VALID_ACTIONS)
    results = []

    for scenario in scenarios:
        action = random.choice(actions)
        output = {
            "reasoning": "Random selection",
            "intent_assessment": random.choice(intents),
            "missing_information": [],
            "action": action,
            "action_detail": f"Random {action.lower()} action",
        }
        results.append({
            "scenario_id": scenario.scenario_id,
            "raw_text": json.dumps(output),
            "parsed": output,
            "model_output": output,
            "valid": True,
        })

    return results


def run_model_inference(
    scenarios, model, tokenizer, system_prompt, temperature, device,
    condition_name,
):
    """Run model inference over all scenarios."""
    runner = PolicyRunner(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_new_tokens=512,
        temperature=temperature,
        device=device,
    )

    results = []
    n = len(scenarios)
    for i, scenario in enumerate(scenarios):
        print(
            f"  [{condition_name}] {i + 1}/{n}: "
            f"{scenario.scenario_id}",
            end="\r",
        )
        try:
            parsed, raw = runner.predict(scenario.state)
            output = parse_model_output(parsed) if parsed else None
            results.append({
                "scenario_id": scenario.scenario_id,
                "raw_text": raw,
                "parsed": parsed,
                "model_output": (
                    output.to_dict()
                    if output and output.is_valid else None
                ),
                "valid": output is not None and output.is_valid,
            })
        except Exception as e:
            print(f"\n  Error on {scenario.scenario_id}: {e}")
            results.append({
                "scenario_id": scenario.scenario_id,
                "raw_text": str(e),
                "parsed": None,
                "model_output": None,
                "valid": False,
            })

    print()  # clear carriage return line
    return results


def print_summary(metrics: dict, condition: str, split: str):
    """Print a summary of key metrics."""
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {condition.upper()} on {split.upper()}")
    print("=" * 60)

    rows = [
        ("Scenarios evaluated", metrics["n_scenarios"]),
        ("Valid outputs", metrics["n_valid_outputs"]),
        ("Format compliance", f"{metrics['format_compliance']:.4f}"),
        ("Composite reward (mean)", f"{metrics['composite_reward_mean']:.4f}"),
        ("Composite reward (std)", f"{metrics['composite_reward_std']:.4f}"),
        ("Intent accuracy", f"{metrics['intent_accuracy']:.4f}"),
        ("Action accuracy (exact)", f"{metrics['action_accuracy']:.4f}"),
        (
            "Action accuracy (acceptable)",
            f"{metrics['action_accuracy_acceptable']:.4f}",
        ),
        ("Macro action F1", f"{metrics['macro_action_f1']:.4f}"),
        ("Action entropy", f"{metrics['action_entropy']:.4f}"),
        (
            "Missing info recall",
            f"{metrics['missing_info_recall_mean']:.4f}",
        ),
        (
            "Missing info precision",
            f"{metrics['missing_info_precision_mean']:.4f}",
        ),
        (
            "Unnecessary question rate",
            f"{metrics['unnecessary_question_rate']:.4f}",
        ),
    ]

    for label, value in rows:
        print(f"  {label:<35} {value}")

    # Per-domain reward
    per_domain = metrics.get("per_domain_reward", {})
    if per_domain:
        print(f"\n{'Per-domain reward:'}")
        for domain, reward in sorted(
            per_domain.items(), key=lambda x: -x[1]
        ):
            print(f"    {domain:<30} {reward:.4f}")

    # Per-action metrics
    per_action = metrics.get("per_action_metrics", {})
    if per_action:
        print(f"\n{'Per-action P/R/F1:'}")
        print(
            f"  {'Action':<25} {'Prec':>8} {'Recall':>8} "
            f"{'F1':>8} {'Support':>8}"
        )
        print("  " + "-" * 57)
        for action in sorted(per_action.keys()):
            m = per_action[action]
            print(
                f"  {action:<25} {m['precision']:>8.4f} "
                f"{m['recall']:>8.4f} {m['f1']:>8.4f} "
                f"{m['support']:>8d}"
            )


def print_parity_table(condition, split, temperature, reward_weights_path):
    """Print parity table showing all key evaluation settings."""
    print("\n" + "=" * 60)
    print("PARITY TABLE (evaluation settings)")
    print("=" * 60)

    config = CONDITION_CONFIG[condition]
    adapter = config["adapter_path"] or "None"
    prompt_type = (
        "MINIMAL" if config["system_prompt"] is SYSTEM_PROMPT_MINIMAL
        else "ENGINEERED"
        if config["system_prompt"] is SYSTEM_PROMPT_ENGINEERED
        else "None"
    )

    rows = [
        ("Condition", condition),
        ("Split", split),
        ("Base model", MODEL_PATH),
        ("Adapter path", adapter),
        ("System prompt", prompt_type),
        ("Temperature", temperature),
        ("max_new_tokens", 512),
        ("top_p", 0.9),
        ("Reward weights", reward_weights_path or "default"),
        ("Random seed", SEED),
        ("Timestamp", datetime.now().isoformat(timespec="seconds")),
    ]

    for label, value in rows:
        print(f"  {label:<25} {value}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    condition = args.condition
    split = args.split
    temperature = args.temperature
    output_dir = args.output_dir
    reward_weights_path = args.reward_weights

    os.makedirs(output_dir, exist_ok=True)

    config = CONDITION_CONFIG[condition]

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {config['description']}")
    print(f"Split: {split}")
    print(f"Temperature: {temperature}")
    if reward_weights_path:
        print(f"Reward weights: {reward_weights_path}")
    print(f"{'=' * 60}\n")

    # --- Load scenarios ---
    scenarios = load_scenarios(split)
    print(f"Loaded {len(scenarios)} scenarios from {split} split")

    # --- Set up reward calculator ---
    if reward_weights_path:
        reward_calc = RewardCalculator.from_yaml(reward_weights_path)
        print(f"Using custom reward weights from {reward_weights_path}")
    else:
        reward_calc = RewardCalculator()
        print("Using default reward weights")

    # --- Determine device ---
    device = get_device()
    print(f"Device: {device}")

    # --- Run evaluation ---
    t0 = time.time()

    if condition == "random":
        print("\nRunning random baseline...")
        results = run_random_baseline(scenarios)
    else:
        model, tokenizer = load_model_and_tokenizer(condition, device)
        print(f"\nRunning inference ({condition})...")
        results = run_model_inference(
            scenarios=scenarios,
            model=model,
            tokenizer=tokenizer,
            system_prompt=config["system_prompt"],
            temperature=temperature,
            device=device,
            condition_name=condition,
        )

    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.1f}s")

    # --- Compute metrics ---
    print("\nComputing metrics...")
    metrics = compute_metrics(results, scenarios, reward_calc)
    metrics["inference_time_sec"] = round(elapsed, 1)
    metrics["condition"] = condition
    metrics["split"] = split
    metrics["temperature"] = temperature
    metrics["reward_weights"] = reward_weights_path or "default"

    # --- Print confusion matrix ---
    print("\nConfusion Matrix:")
    print_confusion_matrix(metrics)

    # --- Print summary ---
    print_summary(metrics, condition, split)

    # --- Print parity table ---
    print_parity_table(condition, split, temperature, reward_weights_path)

    # --- Save results ---
    results_path = os.path.join(
        output_dir, f"{condition}_{split}_results.json"
    )
    predictions_path = os.path.join(
        output_dir, f"{condition}_{split}_predictions.json"
    )

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {results_path}")

    with open(predictions_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
