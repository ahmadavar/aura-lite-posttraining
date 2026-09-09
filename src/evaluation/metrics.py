"""Evaluation metrics with bootstrap confidence intervals."""

import json
import math
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

from src.environment.state import Scenario
from src.environment.actions import (
    parse_model_output,
    VALID_ACTIONS,
    ActionType,
)
from src.rewards.reward import RewardCalculator

# Sorted action list for stable ordering in confusion matrix / display
ACTION_LIST = sorted(VALID_ACTIONS)


def compute_metrics(
    results: List[dict],
    scenarios: List[Scenario],
    reward_calc: RewardCalculator,
) -> Dict:
    """Compute all evaluation metrics from prediction results.

    Args:
        results: List of dicts from PolicyRunner.predict_batch
        scenarios: Corresponding scenario objects
        reward_calc: Reward calculator instance

    Returns:
        Dict with all metrics
    """
    scenario_map = {s.scenario_id: s for s in scenarios}

    rewards = []
    intent_correct = []
    action_correct = []
    action_acceptable = []
    missing_info_recalls = []
    missing_info_precisions = []
    unnecessary_asks = []
    output_lengths = []
    valid_outputs = 0
    action_counts = Counter()
    per_domain = {}

    # Confusion matrix: {true_action: {predicted_action: count}}
    confusion = {
        a: {b: 0 for b in ACTION_LIST} for a in ACTION_LIST
    }

    # Track true/predicted action pairs for per-class metrics
    true_actions = []
    pred_actions = []

    for r in results:
        sid = r["scenario_id"]
        scenario = scenario_map[sid]
        gt = scenario.ground_truth

        if not r.get("valid") or not r.get("parsed"):
            rewards.append(0.0)
            intent_correct.append(0)
            action_correct.append(0)
            action_acceptable.append(0)
            continue

        valid_outputs += 1
        output = parse_model_output(r["parsed"])
        if output is None or not output.is_valid:
            rewards.append(0.0)
            continue

        # Compute reward
        reward, breakdown = reward_calc.compute(output, gt)
        rewards.append(reward)

        # Intent accuracy
        pred_intent = output.intent_assessment.strip().lower()
        true_intent = gt.true_intent.strip().lower()
        intent_correct.append(1 if pred_intent == true_intent else 0)

        # Action accuracy (exact match with ideal)
        action_correct.append(
            1 if output.action == gt.ideal_action else 0
        )
        action_counts[output.action] += 1

        # Acceptable action accuracy
        action_acceptable.append(
            1 if output.action in gt.acceptable_actions else 0
        )

        # Confusion matrix tracking
        true_act = gt.ideal_action
        pred_act = output.action
        if true_act in VALID_ACTIONS and pred_act in VALID_ACTIONS:
            confusion[true_act][pred_act] += 1
        true_actions.append(true_act)
        pred_actions.append(pred_act)

        # Missing info recall
        if gt.required_information:
            required = {i.lower() for i in gt.required_information}
            predicted = {i.lower() for i in output.missing_information}
            hits = sum(
                1 for p in predicted
                for req in required
                if p in req or req in p
            )
            recall = min(1.0, hits / len(required))
            missing_info_recalls.append(recall)

        # Missing info precision
        if output.missing_information:
            predicted = {i.lower() for i in output.missing_information}
            if gt.required_information:
                required = {
                    i.lower() for i in gt.required_information
                }
                tp = sum(
                    1 for p in predicted
                    for req in required
                    if p in req or req in p
                )
                precision = min(1.0, tp / len(predicted))
            else:
                # Everything predicted is a false positive
                precision = 0.0
            missing_info_precisions.append(precision)

        # Unnecessary ask detection
        if output.action == "ASK_CLARIFICATION":
            if not gt.required_information:
                unnecessary_asks.append(1)
            else:
                unnecessary_asks.append(0)

        # Output length
        output_lengths.append(len(r.get("raw_text", "")))

        # Per-domain tracking
        domain = gt.true_intent
        if domain not in per_domain:
            per_domain[domain] = []
        per_domain[domain].append(reward)

    # Per-action precision, recall, F1 from confusion matrix
    per_action = _compute_per_action_metrics(confusion)

    # Macro F1: average F1 across classes that appear in the test set
    present_classes = {a for a in ACTION_LIST if any(
        confusion[a][b] > 0 for b in ACTION_LIST
    ) or any(
        confusion[b][a] > 0 for b in ACTION_LIST
    )}
    f1_values = [
        per_action[a]["f1"] for a in present_classes
        if a in per_action
    ]
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    # Action entropy: H = -sum(p * log2(p)) over predicted distribution
    total_preds = sum(action_counts.values())
    action_ent = 0.0
    if total_preds > 0:
        for act, cnt in action_counts.items():
            p = cnt / total_preds
            if p > 0:
                action_ent -= p * math.log2(p)

    n = len(results)
    metrics = {
        "n_scenarios": n,
        "n_valid_outputs": valid_outputs,
        "format_compliance": valid_outputs / n if n > 0 else 0,
        "composite_reward_mean": float(np.mean(rewards)),
        "composite_reward_std": float(np.std(rewards)),
        "intent_accuracy": (
            float(np.mean(intent_correct))
            if intent_correct else 0
        ),
        "action_accuracy": (
            float(np.mean(action_correct))
            if action_correct else 0
        ),
        "action_accuracy_acceptable": (
            float(np.mean(action_acceptable))
            if action_acceptable else 0
        ),
        "missing_info_recall_mean": (
            float(np.mean(missing_info_recalls))
            if missing_info_recalls else 0
        ),
        "missing_info_precision_mean": (
            float(np.mean(missing_info_precisions))
            if missing_info_precisions else 0
        ),
        "unnecessary_question_rate": (
            float(np.mean(unnecessary_asks))
            if unnecessary_asks else 0
        ),
        "avg_output_tokens": (
            float(np.mean(output_lengths))
            if output_lengths else 0
        ),
        "action_distribution": dict(action_counts),
        "action_confusion_matrix": confusion,
        "per_action_metrics": per_action,
        "macro_action_f1": round(macro_f1, 4),
        "action_entropy": round(action_ent, 4),
        "per_domain_reward": {
            k: round(float(np.mean(v)), 4)
            for k, v in per_domain.items()
        },
    }

    return metrics


def _compute_per_action_metrics(
    confusion: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 for each action from confusion matrix.

    Args:
        confusion: {true_action: {predicted_action: count}}

    Returns:
        {action: {"precision": float, "recall": float, "f1": float}}
    """
    per_action = {}
    for action in ACTION_LIST:
        # True positives: confusion[action][action]
        tp = confusion[action][action]

        # False positives: other true classes predicted as this action
        fp = sum(
            confusion[other][action]
            for other in ACTION_LIST
            if other != action
        )

        # False negatives: this true class predicted as other actions
        fn = sum(
            confusion[action][other]
            for other in ACTION_LIST
            if other != action
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_action[action] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return per_action


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns: (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    means = sorted(means)
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    return (
        float(np.mean(arr)),
        float(means[lower_idx]),
        float(means[upper_idx]),
    )


def compare_models(
    baseline_metrics: Dict,
    trained_metrics: Dict,
) -> Dict:
    """Compare baseline vs post-trained metrics."""
    comparison = {}
    keys = [
        "composite_reward_mean",
        "intent_accuracy",
        "action_accuracy",
        "action_accuracy_acceptable",
        "missing_info_recall_mean",
        "missing_info_precision_mean",
        "unnecessary_question_rate",
        "format_compliance",
        "macro_action_f1",
        "action_entropy",
    ]
    for k in keys:
        base_val = baseline_metrics.get(k, 0)
        trained_val = trained_metrics.get(k, 0)
        delta = trained_val - base_val
        comparison[k] = {
            "baseline": round(base_val, 4),
            "post_trained": round(trained_val, 4),
            "delta": round(delta, 4),
            "improved": delta > 0,
        }
    return comparison


def print_confusion_matrix(metrics: Dict) -> None:
    """Print a readable confusion matrix to the terminal.

    Example output:
                        Predicted
    True         ASK  SEARCH  STEP  CLAIM  ESCAL  COMP
    ASK           45      3     1      0      0     0
    SEARCH_KB      2     10     1      0      0     0
    ...
    """
    confusion = metrics.get("action_confusion_matrix", {})
    if not confusion:
        print("No confusion matrix available.")
        return

    # Short labels for display
    short = {
        "ASK_CLARIFICATION": "ASK",
        "SEARCH_KB": "SEARCH",
        "PROVIDE_STEP": "STEP",
        "ROUTE_CLAIM": "CLAIM",
        "ESCALATE": "ESCAL",
        "COMPLETE": "COMP",
    }

    actions = ACTION_LIST
    labels = [short.get(a, a[:6]) for a in actions]
    col_width = max(len(lb) for lb in labels) + 2

    # Header
    header_pad = " " * 18
    print(f"{header_pad}{'Predicted':^{col_width * len(labels)}}")
    header = " " * 18 + "".join(
        f"{lb:>{col_width}}" for lb in labels
    )
    print(header)

    # Rows
    for true_act in actions:
        row_label = short.get(true_act, true_act[:6])
        row_vals = "".join(
            f"{confusion[true_act][pred_act]:>{col_width}}"
            for pred_act in actions
        )
        print(f"{row_label:<18}{row_vals}")
