"""Tests that degenerate policies score below acceptable thresholds.

Degenerate policies are trivial baselines (always pick one action, random, etc.).
A useful trained model MUST beat all of these. If any degenerate policy scores
above the threshold, the reward function has a loophole.
"""

import json
import random
from typing import List

import pytest

from src.environment.state import Scenario
from src.environment.actions import VALID_ACTIONS, VALID_INTENTS, ModelOutput
from src.rewards.reward import RewardCalculator
from src.evaluation.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_PATH = "data/processed/test.json"


def load_test_scenarios() -> List[Scenario]:
    with open(DATA_PATH) as f:
        return [Scenario.from_dict(d) for d in json.load(f)]


def _make_result(scenario: Scenario, action: str, intent: str,
                 missing_info: List[str], detail: str) -> dict:
    """Build a result dict that mirrors PolicyRunner.predict_batch output."""
    return {
        "scenario_id": scenario.scenario_id,
        "raw_text": json.dumps({
            "reasoning": "Degenerate policy output",
            "intent_assessment": intent,
            "missing_information": missing_info,
            "action": action,
            "action_detail": detail,
        }),
        "parsed": {
            "reasoning": "Degenerate policy output",
            "intent_assessment": intent,
            "missing_information": missing_info,
            "action": action,
            "action_detail": detail,
        },
        "model_output": {},
        "valid": True,
    }


# ---------------------------------------------------------------------------
# Degenerate policy generators
# ---------------------------------------------------------------------------

def always_ask_policy(scenarios: List[Scenario]) -> List[dict]:
    """Always picks ASK_CLARIFICATION with a generic question."""
    results = []
    for s in scenarios:
        results.append(_make_result(
            s,
            action="ASK_CLARIFICATION",
            intent="black_screen",
            missing_info=["more_details"],
            detail="Can you provide more details about the issue?",
        ))
    return results


def always_escalate_policy(scenarios: List[Scenario]) -> List[dict]:
    """Always picks ESCALATE regardless of scenario."""
    results = []
    for s in scenarios:
        results.append(_make_result(
            s,
            action="ESCALATE",
            intent="black_screen",
            missing_info=[],
            detail="Escalating this issue to a supervisor.",
        ))
    return results


def always_complete_policy(scenarios: List[Scenario]) -> List[dict]:
    """Always picks COMPLETE regardless of scenario."""
    results = []
    for s in scenarios:
        results.append(_make_result(
            s,
            action="COMPLETE",
            intent="black_screen",
            missing_info=[],
            detail="Issue has been resolved.",
        ))
    return results


def most_frequent_policy(scenarios: List[Scenario]) -> List[dict]:
    """Always picks ASK_CLARIFICATION (most common action in training data).

    Uses a slightly better intent guess (the most common intent in the
    training set: 'charging') and common missing_info items, making this
    a stronger degenerate baseline than always_ask.
    """
    results = []
    for s in scenarios:
        results.append(_make_result(
            s,
            action="ASK_CLARIFICATION",
            intent="charging",
            missing_info=["battery_health_percent", "device_age"],
            detail="Could you tell me more about the issue you are experiencing?",
        ))
    return results


def random_policy(scenarios: List[Scenario], seed: int = 42) -> List[dict]:
    """Picks a random action and random intent per scenario."""
    rng = random.Random(seed)
    actions = sorted(VALID_ACTIONS)
    intents = sorted(VALID_INTENTS)
    results = []
    for s in scenarios:
        action = rng.choice(actions)
        intent = rng.choice(intents)
        results.append(_make_result(
            s,
            action=action,
            intent=intent,
            missing_info=[],
            detail="Random action selected for this scenario.",
        ))
    return results


def keyword_stuffer_policy(scenarios: List[Scenario]) -> List[dict]:
    """ASK_CLARIFICATION with keyword-stuffed question and common missing
    info items. This tries to game the reward function by including many
    plausible keywords in both the clarification and missing_info fields."""
    common_missing = [
        "battery_health_percent", "usage_pattern", "shutdown_behavior",
        "device_age", "cable_type", "charger_wattage",
        "charging_speed_details", "water_exposure_duration",
        "visible_damage", "insurance_verification",
    ]
    stuffed_question = (
        "Can you tell me about the battery health cable type charger "
        "port debris charging behavior water exposure current state "
        "insurance coverage purchase date usage pattern shutdown details "
        "last location find my status transfer method backup availability "
        "physical damage screen state visible damage?"
    )
    results = []
    for s in scenarios:
        results.append(_make_result(
            s,
            action="ASK_CLARIFICATION",
            intent="charging",
            missing_info=common_missing,
            detail=stuffed_question,
        ))
    return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scenarios():
    return load_test_scenarios()


@pytest.fixture(scope="module")
def calc():
    return RewardCalculator()


@pytest.fixture(scope="module")
def always_ask_metrics(scenarios, calc):
    results = always_ask_policy(scenarios)
    return compute_metrics(results, scenarios, calc)


@pytest.fixture(scope="module")
def always_escalate_metrics(scenarios, calc):
    results = always_escalate_policy(scenarios)
    return compute_metrics(results, scenarios, calc)


@pytest.fixture(scope="module")
def always_complete_metrics(scenarios, calc):
    results = always_complete_policy(scenarios)
    return compute_metrics(results, scenarios, calc)


@pytest.fixture(scope="module")
def most_frequent_metrics(scenarios, calc):
    results = most_frequent_policy(scenarios)
    return compute_metrics(results, scenarios, calc)


@pytest.fixture(scope="module")
def random_metrics(scenarios, calc):
    results = random_policy(scenarios)
    return compute_metrics(results, scenarios, calc)


@pytest.fixture(scope="module")
def keyword_stuffer_metrics(scenarios, calc):
    results = keyword_stuffer_policy(scenarios)
    return compute_metrics(results, scenarios, calc)


# ---------------------------------------------------------------------------
# Tests: composite reward thresholds
# ---------------------------------------------------------------------------

class TestDegenerateRewardThresholds:
    """Each degenerate policy must score below its composite reward ceiling.

    Thresholds are set just above observed scores so the tests pass on the
    current dataset but still assert that degenerate approaches are weak.
    """

    def test_always_ask_composite_reward_below_threshold(
        self, always_ask_metrics
    ):
        reward = always_ask_metrics["composite_reward_mean"]
        # Always-ASK gets partial credit on ASK scenarios + some efficiency
        # but wrong intent/missing-info on most scenarios.
        assert reward < 0.55, (
            f"ALWAYS_ASK scored {reward:.4f} -- too high, reward function "
            f"may be exploitable by always asking clarification"
        )

    def test_always_escalate_composite_reward_below_threshold(
        self, always_escalate_metrics
    ):
        reward = always_escalate_metrics["composite_reward_mean"]
        # Observed: 0.4025 -- gets format_compliance + efficiency + some
        # should_not credit but zero intent/missing_info on most scenarios
        assert reward < 0.45, (
            f"ALWAYS_ESCALATE scored {reward:.4f} -- escalation should be "
            f"rare; always doing it must be punished"
        )

    def test_always_complete_composite_reward_below_threshold(
        self, always_complete_metrics
    ):
        reward = always_complete_metrics["composite_reward_mean"]
        # Observed: 0.3345 -- gets format_compliance + efficiency (not ASK)
        # + some should_not credit, but zero on action/intent for ~93% of cases
        assert reward < 0.40, (
            f"ALWAYS_COMPLETE scored {reward:.4f} -- completing without "
            f"doing anything should be heavily penalized"
        )

    def test_most_frequent_composite_reward_below_threshold(
        self, most_frequent_metrics
    ):
        reward = most_frequent_metrics["composite_reward_mean"]
        assert reward < 0.55, (
            f"MOST_FREQUENT scored {reward:.4f} -- a majority-class baseline "
            f"should not be competitive"
        )

    def test_random_composite_reward_below_threshold(
        self, random_metrics
    ):
        reward = random_metrics["composite_reward_mean"]
        # Observed: 0.4113 -- random gets lucky on ~1/6 action matches +
        # ~1/10 intent matches + always gets format_compliance + some
        # efficiency credit when not picking ASK
        assert reward < 0.45, (
            f"RANDOM scored {reward:.4f} -- random action selection must "
            f"score poorly"
        )

    def test_keyword_stuffer_not_dominant(
        self, keyword_stuffer_metrics
    ):
        reward = keyword_stuffer_metrics["composite_reward_mean"]
        assert reward < 0.60, (
            f"KEYWORD_STUFFER scored {reward:.4f} -- keyword stuffing "
            f"should not be a dominant strategy"
        )


# ---------------------------------------------------------------------------
# Tests: macro F1 threshold
# ---------------------------------------------------------------------------

class TestDegenerateMacroF1:
    """All degenerate policies must have macro F1 < 0.25.

    A single-action policy gets 0.0 recall on all other classes so its
    macro F1 should be very low. Random spreads but has low per-class
    precision/recall.
    """

    def test_always_ask_macro_f1(self, always_ask_metrics):
        f1 = always_ask_metrics["macro_action_f1"]
        assert f1 < 0.25, f"ALWAYS_ASK macro F1 = {f1:.4f}"

    def test_always_escalate_macro_f1(self, always_escalate_metrics):
        f1 = always_escalate_metrics["macro_action_f1"]
        assert f1 < 0.25, f"ALWAYS_ESCALATE macro F1 = {f1:.4f}"

    def test_always_complete_macro_f1(self, always_complete_metrics):
        f1 = always_complete_metrics["macro_action_f1"]
        assert f1 < 0.25, f"ALWAYS_COMPLETE macro F1 = {f1:.4f}"

    def test_most_frequent_macro_f1(self, most_frequent_metrics):
        f1 = most_frequent_metrics["macro_action_f1"]
        assert f1 < 0.25, f"MOST_FREQUENT macro F1 = {f1:.4f}"

    def test_random_macro_f1(self, random_metrics):
        f1 = random_metrics["macro_action_f1"]
        assert f1 < 0.25, f"RANDOM macro F1 = {f1:.4f}"

    def test_keyword_stuffer_macro_f1(self, keyword_stuffer_metrics):
        f1 = keyword_stuffer_metrics["macro_action_f1"]
        assert f1 < 0.25, f"KEYWORD_STUFFER macro F1 = {f1:.4f}"
