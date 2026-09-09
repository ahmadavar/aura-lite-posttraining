#!/usr/bin/env python3
"""Run ALL degenerate policies on the test set and print a comparison table.

Usage:
    PYTHONPATH=. python3.11 scripts/run_degenerate_gate.py

Outputs:
    - Prints a formatted table to stdout
    - Saves detailed results to results/degenerate_gate.json
"""

import json
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.state import Scenario
from src.rewards.reward import RewardCalculator
from src.evaluation.metrics import compute_metrics

# Re-use the policy generators from the test module
from tests.test_degenerate import (
    always_ask_policy,
    always_escalate_policy,
    always_complete_policy,
    most_frequent_policy,
    random_policy,
    keyword_stuffer_policy,
    load_test_scenarios,
)

# Gate threshold: all degenerate policies must score below this
GATE_THRESHOLD = 0.55


def run_gate():
    """Evaluate all degenerate policies and report results."""
    scenarios = load_test_scenarios()
    calc = RewardCalculator()

    policies = {
        "ALWAYS_ASK": always_ask_policy(scenarios),
        "ALWAYS_ESCALATE": always_escalate_policy(scenarios),
        "ALWAYS_COMPLETE": always_complete_policy(scenarios),
        "MOST_FREQUENT": most_frequent_policy(scenarios),
        "RANDOM": random_policy(scenarios),
        "KEYWORD_STUFFER": keyword_stuffer_policy(scenarios),
    }

    all_metrics = {}
    table_rows = []

    for name, results in policies.items():
        metrics = compute_metrics(results, scenarios, calc)
        all_metrics[name] = metrics

        table_rows.append({
            "policy": name,
            "reward": metrics["composite_reward_mean"],
            "action_acc": metrics["action_accuracy"],
            "macro_f1": metrics["macro_action_f1"],
            "uq_rate": metrics["unnecessary_question_rate"],
        })

    # Print table
    print()
    print("DEGENERATE POLICY GATE")
    print("=" * 66)
    print(
        f"{'Policy':<22} {'Reward':>8} {'Act_Acc':>8} "
        f"{'MacroF1':>8} {'UQ_Rate':>8}"
    )
    print("-" * 66)

    gate_pass = True
    for row in table_rows:
        print(
            f"{row['policy']:<22} {row['reward']:>8.4f} "
            f"{row['action_acc']:>8.4f} {row['macro_f1']:>8.4f} "
            f"{row['uq_rate']:>8.4f}"
        )
        if row["reward"] >= GATE_THRESHOLD:
            gate_pass = False

    print("-" * 66)

    if gate_pass:
        print(
            f"GATE: PASS  (all degenerate policies below "
            f"{GATE_THRESHOLD} composite reward)"
        )
    else:
        violators = [
            r["policy"] for r in table_rows
            if r["reward"] >= GATE_THRESHOLD
        ]
        print(
            f"GATE: FAIL  (violators: {', '.join(violators)} >= "
            f"{GATE_THRESHOLD})"
        )

    print()

    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        "gate_threshold": GATE_THRESHOLD,
        "gate_passed": gate_pass,
        "n_scenarios": len(scenarios),
        "policies": {},
    }
    for name, metrics in all_metrics.items():
        # Remove non-serializable numpy types
        output["policies"][name] = {
            "composite_reward_mean": float(metrics["composite_reward_mean"]),
            "composite_reward_std": float(metrics["composite_reward_std"]),
            "intent_accuracy": float(metrics["intent_accuracy"]),
            "action_accuracy": float(metrics["action_accuracy"]),
            "action_accuracy_acceptable": float(
                metrics["action_accuracy_acceptable"]
            ),
            "macro_action_f1": float(metrics["macro_action_f1"]),
            "action_entropy": float(metrics["action_entropy"]),
            "missing_info_recall_mean": float(
                metrics["missing_info_recall_mean"]
            ),
            "missing_info_precision_mean": float(
                metrics["missing_info_precision_mean"]
            ),
            "unnecessary_question_rate": float(
                metrics["unnecessary_question_rate"]
            ),
            "format_compliance": float(metrics["format_compliance"]),
            "action_distribution": {
                k: int(v)
                for k, v in metrics["action_distribution"].items()
            },
            "per_domain_reward": {
                k: float(v)
                for k, v in metrics["per_domain_reward"].items()
            },
        }

    results_path = "results/degenerate_gate.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {results_path}")

    return gate_pass


if __name__ == "__main__":
    passed = run_gate()
    sys.exit(0 if passed else 1)
