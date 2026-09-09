"""Unit tests for reward components and calculator."""

import pytest
from src.environment.actions import ModelOutput
from src.environment.state import GroundTruth
from src.rewards.components import (
    r_intent, r_action, r_missing_info, r_clarification,
    r_efficiency, r_should_not, r_format,
)
from src.rewards.reward import RewardCalculator


def make_gt(**overrides):
    defaults = {
        "true_intent": "black_screen",
        "true_issue": "software_crash",
        "required_information": [
            "physical_damage_check", "recent_events"
        ],
        "ideal_action": "ASK_CLARIFICATION",
        "ideal_clarification": "Did the phone get dropped?",
        "acceptable_actions": ["ASK_CLARIFICATION", "SEARCH_KB"],
        "acceptable_clarification_topics": [
            "physical_damage", "recent_events", "led_status"
        ],
        "resolution_path": "force_restart",
        "should_not_do": ["COMPLETE", "ROUTE_CLAIM"],
    }
    defaults.update(overrides)
    return GroundTruth(**defaults)


def make_output(**overrides):
    defaults = {
        "reasoning": "Customer reports black screen",
        "intent_assessment": "black_screen",
        "missing_information": ["physical_damage_check"],
        "action": "ASK_CLARIFICATION",
        "action_detail": "Was there any physical damage or drop?",
    }
    defaults.update(overrides)
    return ModelOutput(**defaults)


class TestRIntent:
    def test_exact_match(self):
        assert r_intent(make_output(), make_gt()) == 1.0

    def test_wrong_intent(self):
        out = make_output(intent_assessment="charging")
        assert r_intent(out, make_gt()) == 0.0

    def test_case_insensitive(self):
        out = make_output(intent_assessment="BLACK_SCREEN")
        assert r_intent(out, make_gt()) == 1.0


class TestRAction:
    def test_ideal_action(self):
        assert r_action(make_output(), make_gt()) == 1.0

    def test_acceptable_action(self):
        out = make_output(action="SEARCH_KB")
        assert r_action(out, make_gt()) == 0.5

    def test_wrong_action(self):
        out = make_output(action="COMPLETE")
        assert r_action(out, make_gt()) == 0.0


class TestRMissingInfo:
    def test_full_recall(self):
        out = make_output(
            missing_information=[
                "physical_damage_check", "recent_events"
            ]
        )
        score = r_missing_info(out, make_gt())
        assert score == 1.0

    def test_partial_recall(self):
        out = make_output(
            missing_information=["physical_damage_check"]
        )
        score = r_missing_info(out, make_gt())
        assert score == 0.5

    def test_no_recall(self):
        out = make_output(missing_information=[])
        score = r_missing_info(out, make_gt())
        assert score == 0.0

    def test_false_positive_penalty(self):
        out = make_output(
            missing_information=[
                "physical_damage_check",
                "bogus_1", "bogus_2", "bogus_3",
            ]
        )
        score = r_missing_info(out, make_gt())
        assert score < 0.5  # penalized for 3 FPs


class TestRClarification:
    def test_returns_na_for_non_ask(self):
        out = make_output(action="SEARCH_KB")
        assert r_clarification(out, make_gt()) == -1.0

    def test_relevant_question(self):
        out = make_output(
            action_detail="Was there any physical damage recently?"
        )
        score = r_clarification(out, make_gt())
        assert score > 0.0

    def test_irrelevant_question(self):
        out = make_output(
            action_detail="What color is your phone case?"
        )
        score = r_clarification(out, make_gt())
        assert score == 0.0


class TestREfficiency:
    def test_no_penalty_when_info_needed(self):
        assert r_efficiency(make_output(), make_gt()) == 1.0

    def test_penalty_when_no_info_needed(self):
        gt = make_gt(required_information=[])
        out = make_output(action="ASK_CLARIFICATION")
        assert r_efficiency(out, gt) == 0.0

    def test_no_penalty_for_non_ask(self):
        out = make_output(action="PROVIDE_STEP")
        assert r_efficiency(out, make_gt()) == 1.0


class TestRShouldNot:
    def test_allowed_action(self):
        assert r_should_not(make_output(), make_gt()) == 1.0

    def test_forbidden_action(self):
        out = make_output(action="COMPLETE")
        assert r_should_not(out, make_gt()) == 0.0


class TestRFormat:
    def test_valid_output(self):
        assert r_format(make_output()) == 1.0

    def test_invalid_action(self):
        out = ModelOutput(
            reasoning="test",
            intent_assessment="test",
            missing_information=[],
            action="INVALID_ACTION",
            action_detail="test",
        )
        assert r_format(out) == 0.0


class TestRewardCalculator:
    def test_perfect_score(self):
        calc = RewardCalculator()
        out = make_output(
            missing_information=[
                "physical_damage_check", "recent_events"
            ],
            action_detail="Was there physical damage or recent events?",
        )
        reward, breakdown = calc.compute(out, make_gt())
        assert reward > 0.7
        assert breakdown["format_valid"] is True

    def test_zero_on_all_wrong(self):
        calc = RewardCalculator()
        out = make_output(
            intent_assessment="charging",
            missing_information=[],
            action="COMPLETE",
            action_detail="All done",
        )
        reward, breakdown = calc.compute(out, make_gt())
        assert reward < 0.2

    def test_weight_redistribution_when_not_ask(self):
        calc = RewardCalculator()
        out = make_output(
            action="SEARCH_KB",
            action_detail="black screen troubleshooting",
        )
        _, breakdown = calc.compute(out, make_gt())
        assert breakdown["clarification_na"] is True
        # Weights should sum to ~1.0 after redistribution
        total_weight = sum(breakdown["weights_used"].values())
        assert abs(total_weight - 1.0) < 0.01
