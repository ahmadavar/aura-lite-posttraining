#!/usr/bin/env python3
"""Run baseline evaluations: zero-shot, engineered prompt, random.

Usage:
    PYTHONPATH=. python3.11 scripts/run_baselines.py
"""

import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.environment.state import Scenario
from src.environment.actions import VALID_ACTIONS, VALID_INTENTS
from src.policies.prompt_templates import (
    SYSTEM_PROMPT_MINIMAL,
    SYSTEM_PROMPT_ENGINEERED,
    build_messages,
)
from src.policies.inference import PolicyRunner, extract_json_from_text
from src.rewards.reward import RewardCalculator
from src.evaluation.metrics import compute_metrics, bootstrap_ci


SEED = 42
random.seed(SEED)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "qwen2.5-1.5b-instruct"
)
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "test.json"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results"
)


def load_test_scenarios():
    with open(DATA_PATH) as f:
        data = json.load(f)
    return [Scenario.from_dict(d) for d in data]


def run_random_baseline(scenarios, reward_calc):
    """Baseline C: random action selection."""
    results = []
    intents = list(VALID_INTENTS)
    actions = list(VALID_ACTIONS)

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


def run_model_baseline(
    scenarios, model, tokenizer, system_prompt, name, device="mps"
):
    """Run model baseline with given system prompt."""
    runner = PolicyRunner(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_new_tokens=512,
        temperature=0.3,  # Lower for more deterministic baselines
        device=device,
    )

    results = []
    for i, scenario in enumerate(scenarios):
        print(f"  [{name}] {i+1}/{len(scenarios)}: "
              f"{scenario.scenario_id}", end="\r")
        try:
            parsed, raw = runner.predict(scenario.state)
            from src.environment.actions import parse_model_output
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

    print()
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scenarios = load_test_scenarios()
    reward_calc = RewardCalculator()

    print(f"Loaded {len(scenarios)} test scenarios")

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- Baseline C: Random ---
    print("\n=== Baseline C: Random ===")
    random_results = run_random_baseline(scenarios, reward_calc)
    random_metrics = compute_metrics(
        random_results, scenarios, reward_calc
    )
    print(f"  Composite reward: "
          f"{random_metrics['composite_reward_mean']:.4f}")
    print(f"  Intent accuracy: "
          f"{random_metrics['intent_accuracy']:.4f}")
    print(f"  Action accuracy: "
          f"{random_metrics['action_accuracy']:.4f}")

    # --- Load model ---
    print(f"\nLoading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    # --- Baseline A: Zero-shot ---
    print("\n=== Baseline A: Zero-shot (minimal prompt) ===")
    t0 = time.time()
    zeroshot_results = run_model_baseline(
        scenarios, model, tokenizer,
        SYSTEM_PROMPT_MINIMAL, "zero-shot", device
    )
    t_zeroshot = time.time() - t0
    zeroshot_metrics = compute_metrics(
        zeroshot_results, scenarios, reward_calc
    )
    zeroshot_metrics["inference_time_sec"] = round(t_zeroshot, 1)
    print(f"  Composite reward: "
          f"{zeroshot_metrics['composite_reward_mean']:.4f}")
    print(f"  Intent accuracy: "
          f"{zeroshot_metrics['intent_accuracy']:.4f}")
    print(f"  Action accuracy: "
          f"{zeroshot_metrics['action_accuracy']:.4f}")
    print(f"  Format compliance: "
          f"{zeroshot_metrics['format_compliance']:.4f}")
    print(f"  Time: {t_zeroshot:.1f}s")

    # --- Baseline B: Engineered prompt ---
    print("\n=== Baseline B: Engineered prompt ===")
    t0 = time.time()
    engineered_results = run_model_baseline(
        scenarios, model, tokenizer,
        SYSTEM_PROMPT_ENGINEERED, "engineered", device
    )
    t_eng = time.time() - t0
    engineered_metrics = compute_metrics(
        engineered_results, scenarios, reward_calc
    )
    engineered_metrics["inference_time_sec"] = round(t_eng, 1)
    print(f"  Composite reward: "
          f"{engineered_metrics['composite_reward_mean']:.4f}")
    print(f"  Intent accuracy: "
          f"{engineered_metrics['intent_accuracy']:.4f}")
    print(f"  Action accuracy: "
          f"{engineered_metrics['action_accuracy']:.4f}")
    print(f"  Format compliance: "
          f"{engineered_metrics['format_compliance']:.4f}")
    print(f"  Time: {t_eng:.1f}s")

    # --- Save all results ---
    all_baselines = {
        "random": {
            "metrics": random_metrics,
            "predictions": random_results,
        },
        "zero_shot": {
            "metrics": zeroshot_metrics,
            "predictions": zeroshot_results,
        },
        "engineered_prompt": {
            "metrics": engineered_metrics,
            "predictions": engineered_results,
        },
    }

    with open(
        os.path.join(RESULTS_DIR, "baseline_results.json"), "w"
    ) as f:
        json.dump(all_baselines, f, indent=2, default=str)

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Random':>10} {'Zero-shot':>10} "
          f"{'Engineered':>10}")
    print("-" * 60)
    for key in [
        "composite_reward_mean", "intent_accuracy",
        "action_accuracy", "format_compliance",
        "missing_info_recall_mean", "unnecessary_question_rate",
    ]:
        r = random_metrics.get(key, 0)
        z = zeroshot_metrics.get(key, 0)
        e = engineered_metrics.get(key, 0)
        print(f"{key:<30} {r:>10.4f} {z:>10.4f} {e:>10.4f}")

    print(f"\nResults saved to {RESULTS_DIR}/baseline_results.json")


if __name__ == "__main__":
    main()
