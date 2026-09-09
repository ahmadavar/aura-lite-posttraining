"""Main evaluation script -- runs model on test set and computes metrics."""

import json
import os
import sys
from typing import List

from src.environment.state import Scenario
from src.rewards.reward import RewardCalculator
from src.evaluation.metrics import (
    compute_metrics,
    bootstrap_ci,
    compare_models,
)


def load_scenarios(path: str) -> List[Scenario]:
    """Load scenarios from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [Scenario.from_dict(d) for d in data]


def evaluate_predictions(
    predictions_path: str,
    scenarios_path: str,
    reward_weights_path: str = None,
    output_path: str = None,
) -> dict:
    """Evaluate saved predictions against scenarios.

    Args:
        predictions_path: Path to JSON with prediction results
        scenarios_path: Path to test scenarios JSON
        reward_weights_path: Optional path to reward weights YAML
        output_path: Optional path to save metrics JSON
    """
    scenarios = load_scenarios(scenarios_path)

    if reward_weights_path:
        reward_calc = RewardCalculator.from_yaml(reward_weights_path)
    else:
        reward_calc = RewardCalculator()

    with open(predictions_path) as f:
        predictions = json.load(f)

    metrics = compute_metrics(predictions, scenarios, reward_calc)

    # Add bootstrap CI for composite reward
    rewards = []
    scenario_map = {s.scenario_id: s for s in scenarios}
    for pred in predictions:
        if pred.get("valid") and pred.get("parsed"):
            from src.environment.actions import parse_model_output
            output = parse_model_output(pred["parsed"])
            if output and output.is_valid:
                gt = scenario_map[pred["scenario_id"]].ground_truth
                r, _ = reward_calc.compute(output, gt)
                rewards.append(r)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    if rewards:
        mean, lower, upper = bootstrap_ci(rewards)
        metrics["composite_reward_ci"] = {
            "mean": round(mean, 4),
            "lower_95": round(lower, 4),
            "upper_95": round(upper, 4),
        }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics
