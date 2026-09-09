"""Discrete action space for the customer support decision environment."""

from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    SEARCH_KB = "SEARCH_KB"
    PROVIDE_STEP = "PROVIDE_STEP"
    ROUTE_CLAIM = "ROUTE_CLAIM"
    ESCALATE = "ESCALATE"
    COMPLETE = "COMPLETE"


VALID_ACTIONS = {a.value for a in ActionType}

VALID_INTENTS = {
    "black_screen",
    "charging",
    "battery_drain",
    "activation",
    "wifi_network",
    "data_transfer",
    "damaged_screen",
    "lost_stolen",
    "liquid_damage",
    "claim_routing",
}


class ModelOutput:
    """Parsed and validated model output."""

    def __init__(
        self,
        reasoning: str,
        intent_assessment: str,
        missing_information: list,
        action: str,
        action_detail: str,
    ):
        self.reasoning = reasoning
        self.intent_assessment = intent_assessment
        self.missing_information = missing_information
        self.action = action
        self.action_detail = action_detail
        self.is_valid = self._validate()

    def _validate(self) -> bool:
        if self.action not in VALID_ACTIONS:
            return False
        if not isinstance(self.missing_information, list):
            return False
        if not self.action_detail or not self.action_detail.strip():
            return False
        if not self.reasoning or not self.reasoning.strip():
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "reasoning": self.reasoning,
            "intent_assessment": self.intent_assessment,
            "missing_information": self.missing_information,
            "action": self.action,
            "action_detail": self.action_detail,
        }


def parse_model_output(raw: dict) -> Optional[ModelOutput]:
    """Parse raw dict into validated ModelOutput. Returns None if
    required fields are missing."""
    try:
        return ModelOutput(
            reasoning=raw.get("reasoning", ""),
            intent_assessment=raw.get("intent_assessment", ""),
            missing_information=raw.get("missing_information", []),
            action=raw.get("action", ""),
            action_detail=raw.get("action_detail", ""),
        )
    except (TypeError, AttributeError):
        return None
