"""Episode runner for the contextual bandit environment."""

from typing import Optional, Tuple

from src.environment.state import Scenario, SupportState
from src.environment.actions import ModelOutput, parse_model_output
from src.rewards.reward import RewardCalculator


class Episode:
    """Single-step contextual bandit episode.

    Flow:
    1. Present state to model (no ground truth)
    2. Model produces structured JSON output
    3. Reward function scores output against hidden ground truth
    4. Episode terminates
    """

    def __init__(
        self, scenario: Scenario, reward_calc: RewardCalculator
    ):
        self.scenario = scenario
        self.reward_calc = reward_calc
        self.done = False
        self.model_output: Optional[ModelOutput] = None
        self.reward: Optional[float] = None
        self.reward_breakdown: Optional[dict] = None

    def get_state(self) -> SupportState:
        """Return observable state. Ground truth is never exposed."""
        return self.scenario.state

    def step(self, raw_output: dict) -> Tuple[float, dict]:
        """Process model output and compute reward."""
        self.model_output = parse_model_output(raw_output)

        if self.model_output is None or not self.model_output.is_valid:
            self.reward = 0.0
            self.reward_breakdown = {
                "total": 0.0,
                "format_valid": False,
                "error": "invalid_output",
            }
        else:
            self.reward, self.reward_breakdown = (
                self.reward_calc.compute(
                    self.model_output, self.scenario.ground_truth
                )
            )

        self.done = True
        return self.reward, self.reward_breakdown


def run_episode(
    scenario: Scenario,
    reward_calc: RewardCalculator,
    raw_model_output: dict,
) -> dict:
    """Run a complete episode and return results dict."""
    ep = Episode(scenario, reward_calc)
    reward, breakdown = ep.step(raw_model_output)
    return {
        "scenario_id": scenario.scenario_id,
        "reward": reward,
        "breakdown": breakdown,
        "model_output": (
            ep.model_output.to_dict() if ep.model_output else None
        ),
    }
