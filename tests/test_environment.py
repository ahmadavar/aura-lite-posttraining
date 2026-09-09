"""Unit tests for environment state and episode logic."""

import pytest
from src.environment.state import (
    SupportState, GroundTruth, Scenario
)
from src.environment.actions import (
    ModelOutput, parse_model_output, VALID_ACTIONS
)
from src.environment.episode import Episode, run_episode
from src.rewards.reward import RewardCalculator


def make_scenario():
    return Scenario(
        scenario_id="test_001",
        state=SupportState(
            customer_message="My screen is black",
            device_type="iPhone 14",
            conversation_turn=0,
            available_kb_topics=["force_restart"],
        ),
        ground_truth=GroundTruth(
            true_intent="black_screen",
            true_issue="software_crash",
            required_information=["physical_damage_check"],
            ideal_action="ASK_CLARIFICATION",
            ideal_clarification="Any physical damage?",
            acceptable_actions=["ASK_CLARIFICATION", "SEARCH_KB"],
            acceptable_clarification_topics=["physical_damage"],
            resolution_path="force_restart",
            should_not_do=["COMPLETE", "ROUTE_CLAIM"],
        ),
    )


class TestSupportState:
    def test_to_dict(self):
        state = SupportState(
            customer_message="test",
            device_type="iPhone",
        )
        d = state.to_dict()
        assert d["customer_message"] == "test"
        assert d["device_type"] == "iPhone"
        assert d["conversation_turn"] == 0

    def test_defaults(self):
        state = SupportState(customer_message="test")
        assert state.information_collected == []
        assert state.available_kb_topics == []


class TestScenario:
    def test_roundtrip(self):
        s = make_scenario()
        d = s.to_dict()
        s2 = Scenario.from_dict(d)
        assert s2.scenario_id == s.scenario_id
        assert s2.state.customer_message == s.state.customer_message
        assert s2.ground_truth.true_intent == s.ground_truth.true_intent

    def test_ground_truth_not_in_state(self):
        s = make_scenario()
        state_dict = s.state.to_dict()
        assert "true_intent" not in state_dict
        assert "hidden_ground_truth" not in state_dict
        assert "should_not_do" not in state_dict


class TestModelOutput:
    def test_valid_output(self):
        out = ModelOutput(
            reasoning="test",
            intent_assessment="black_screen",
            missing_information=["damage"],
            action="ASK_CLARIFICATION",
            action_detail="Any damage?",
        )
        assert out.is_valid is True

    def test_invalid_action(self):
        out = ModelOutput(
            reasoning="test",
            intent_assessment="black_screen",
            missing_information=[],
            action="INVALID",
            action_detail="test",
        )
        assert out.is_valid is False

    def test_empty_detail(self):
        out = ModelOutput(
            reasoning="test",
            intent_assessment="black_screen",
            missing_information=[],
            action="ASK_CLARIFICATION",
            action_detail="",
        )
        assert out.is_valid is False


class TestParseModelOutput:
    def test_valid_dict(self):
        raw = {
            "reasoning": "test",
            "intent_assessment": "charging",
            "missing_information": ["charger_type"],
            "action": "SEARCH_KB",
            "action_detail": "charging troubleshoot",
        }
        out = parse_model_output(raw)
        assert out is not None
        assert out.action == "SEARCH_KB"

    def test_missing_fields(self):
        raw = {"reasoning": "test"}
        out = parse_model_output(raw)
        assert out is not None
        assert out.is_valid is False

    def test_none_input(self):
        assert parse_model_output(None) is None


class TestEpisode:
    def test_valid_episode(self):
        scenario = make_scenario()
        calc = RewardCalculator()
        ep = Episode(scenario, calc)

        assert ep.get_state().customer_message == "My screen is black"

        raw = {
            "reasoning": "Screen is black, need to check damage",
            "intent_assessment": "black_screen",
            "missing_information": ["physical_damage_check"],
            "action": "ASK_CLARIFICATION",
            "action_detail": "Was there any physical damage?",
        }
        reward, breakdown = ep.step(raw)
        assert ep.done is True
        assert reward > 0
        assert breakdown["format_valid"] is True

    def test_invalid_output_gets_zero(self):
        scenario = make_scenario()
        calc = RewardCalculator()
        ep = Episode(scenario, calc)

        reward, breakdown = ep.step({"garbage": True})
        assert reward == 0.0
        assert breakdown["error"] == "invalid_output"

    def test_run_episode_convenience(self):
        scenario = make_scenario()
        calc = RewardCalculator()
        raw = {
            "reasoning": "test",
            "intent_assessment": "black_screen",
            "missing_information": [],
            "action": "SEARCH_KB",
            "action_detail": "black screen fix",
        }
        result = run_episode(scenario, calc, raw)
        assert result["scenario_id"] == "test_001"
        assert "reward" in result
        assert "breakdown" in result
