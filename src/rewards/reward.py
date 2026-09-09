"""Compositional reward calculator combining all components."""

from typing import Dict, Tuple
import yaml

from src.environment.actions import ModelOutput
from src.environment.state import GroundTruth
from src.rewards import components


DEFAULT_WEIGHTS = {
    "intent_accuracy": 0.20,
    "action_correctness": 0.25,
    "missing_info_recall": 0.15,
    "clarification_quality": 0.15,
    "efficiency": 0.10,
    "should_not_penalty": 0.10,
    "format_compliance": 0.05,
}


class RewardCalculator:
    """Compute composite reward from individual components.

    When R_clarification is N/A (action != ASK_CLARIFICATION),
    its weight is redistributed proportionally to other components.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    @classmethod
    def from_yaml(cls, path: str) -> "RewardCalculator":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(weights=config.get("reward_weights", {}))

    def compute(
        self,
        output: ModelOutput,
        gt: GroundTruth,
    ) -> Tuple[float, Dict]:
        """Compute total reward and per-component breakdown."""
        # Compute raw component scores
        scores = {
            "intent_accuracy": components.r_intent(output, gt),
            "action_correctness": components.r_action(output, gt),
            "missing_info_recall": components.r_missing_info(
                output, gt
            ),
            "clarification_quality": components.r_clarification(
                output, gt
            ),
            "efficiency": components.r_efficiency(output, gt),
            "should_not_penalty": components.r_should_not(output, gt),
            "format_compliance": components.r_format(output),
        }

        # Handle N/A clarification (redistribute weight)
        clarification_na = scores["clarification_quality"] == -1.0
        active_weights = self.weights.copy()

        if clarification_na:
            scores["clarification_quality"] = 0.0  # zero out
            clar_weight = active_weights.pop(
                "clarification_quality", 0.0
            )
            # Redistribute proportionally
            remaining_sum = sum(active_weights.values())
            if remaining_sum > 0:
                for k in active_weights:
                    active_weights[k] += (
                        clar_weight
                        * active_weights[k]
                        / remaining_sum
                    )

        # Compute weighted total
        total = 0.0
        for component, weight in active_weights.items():
            total += weight * scores.get(component, 0.0)

        breakdown = {
            "total": round(total, 4),
            "format_valid": True,
            "clarification_na": clarification_na,
            "components": {
                k: round(v, 4) for k, v in scores.items()
            },
            "weights_used": {
                k: round(v, 4) for k, v in active_weights.items()
            },
        }

        return round(total, 4), breakdown
