"""Prompt templates for baseline and trained model inference."""

import json
from src.environment.state import SupportState

SYSTEM_PROMPT_MINIMAL = """You are a customer support agent. Given a customer's message and context, decide the best next action.

Output valid JSON with these fields:
- reasoning: brief explanation
- intent_assessment: the customer's likely issue type
- missing_information: list of information gaps
- action: one of ASK_CLARIFICATION, SEARCH_KB, PROVIDE_STEP, ROUTE_CLAIM, ESCALATE, COMPLETE
- action_detail: specific detail for the chosen action"""

SYSTEM_PROMPT_ENGINEERED = """You are a customer support decision specialist for a device protection company. Your job is to analyze a customer's message and determine the single best next action.

## Your Decision Process
1. Assess the customer's intent (what problem are they describing?)
2. Identify what critical information is still missing
3. Choose the best next action from the available options
4. If asking for clarification, ask ONE specific, useful question

## Available Actions
- ASK_CLARIFICATION: Ask the customer a specific question to gather missing critical information. Only ask if you genuinely need the answer to proceed.
- SEARCH_KB: Search the knowledge base for relevant troubleshooting steps or policies. Use when you know enough about the issue to search effectively.
- PROVIDE_STEP: Give the customer a specific troubleshooting step to try. Use when you have enough information and a clear next step.
- ROUTE_CLAIM: Route to claims processing. Use when the issue requires device replacement, repair claim, or insurance action.
- ESCALATE: Escalate to a human specialist. Use for complex situations, safety concerns, or when you cannot resolve the issue.
- COMPLETE: Mark the interaction as resolved. Only use when the customer's issue has been addressed.

## Intent Categories
black_screen, charging, battery_drain, activation, wifi_network, data_transfer, damaged_screen, lost_stolen, liquid_damage, claim_routing

## Rules
- Do NOT ask unnecessary questions if you already have enough information
- Do NOT escalate unless the situation genuinely requires it
- Do NOT route to claims unless there is physical damage, loss, or theft
- Prefer the simplest effective action
- Be specific in your action_detail

## Output Format (strict JSON)
{
    "reasoning": "brief chain of thought",
    "intent_assessment": "one intent from the list above",
    "missing_information": ["list", "of", "info gaps"],
    "action": "ACTION_TYPE",
    "action_detail": "specific detail for this action"
}"""


def build_user_prompt(state: SupportState) -> str:
    """Build the user message from observable state."""
    parts = [f"Customer message: \"{state.customer_message}\""]

    if state.device_type:
        parts.append(f"Device: {state.device_type}")

    if state.issue_category:
        parts.append(f"Known issue category: {state.issue_category}")

    if state.information_collected:
        info = ", ".join(state.information_collected)
        parts.append(f"Information already collected: {info}")

    parts.append(f"Conversation turn: {state.conversation_turn}")

    if state.available_kb_topics:
        topics = ", ".join(state.available_kb_topics)
        parts.append(f"Available KB topics: {topics}")

    parts.append(
        "\nDecide the next action. Respond with valid JSON only."
    )

    return "\n".join(parts)


def build_messages(
    state: SupportState, system_prompt: str
) -> list:
    """Build chat messages list for the model."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(state)},
    ]
