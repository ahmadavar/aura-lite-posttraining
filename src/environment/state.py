"""State representation for the customer support environment."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SupportState:
    """Observable state provided to the model. Never includes ground truth."""

    customer_message: str
    device_type: Optional[str] = None
    issue_category: Optional[str] = None
    information_collected: List[str] = field(default_factory=list)
    conversation_turn: int = 0
    available_kb_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "customer_message": self.customer_message,
            "device_type": self.device_type,
            "issue_category": self.issue_category,
            "information_collected": self.information_collected,
            "conversation_turn": self.conversation_turn,
            "available_kb_topics": self.available_kb_topics,
        }


@dataclass
class GroundTruth:
    """Hidden ground truth -- never shown to the model during inference."""

    true_intent: str
    true_issue: str
    required_information: List[str]
    ideal_action: str
    ideal_clarification: str
    acceptable_actions: List[str]
    acceptable_clarification_topics: List[str]
    resolution_path: str
    should_not_do: List[str]

    def to_dict(self) -> dict:
        return {
            "true_intent": self.true_intent,
            "true_issue": self.true_issue,
            "required_information": self.required_information,
            "ideal_action": self.ideal_action,
            "ideal_clarification": self.ideal_clarification,
            "acceptable_actions": self.acceptable_actions,
            "acceptable_clarification_topics": (
                self.acceptable_clarification_topics
            ),
            "resolution_path": self.resolution_path,
            "should_not_do": self.should_not_do,
        }


@dataclass
class Scenario:
    """Complete scenario with observable state and hidden ground truth."""

    scenario_id: str
    state: SupportState
    ground_truth: GroundTruth

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "state": self.state.to_dict(),
            "hidden_ground_truth": self.ground_truth.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        gt_data = d["hidden_ground_truth"]
        state_data = d.get("state", {})
        if not state_data:
            state_data = {
                "customer_message": d.get("customer_message", ""),
                "device_type": d.get("device_type"),
                "issue_category": d.get("issue_category"),
                "information_collected": d.get(
                    "information_collected", []
                ),
                "conversation_turn": d.get("conversation_turn", 0),
                "available_kb_topics": d.get(
                    "available_kb_topics", []
                ),
            }
        return cls(
            scenario_id=d["scenario_id"],
            state=SupportState(**state_data),
            ground_truth=GroundTruth(**gt_data),
        )
