"""Model inference for baseline and trained policies."""

import json
import re
from typing import Dict, List, Optional, Tuple

from src.environment.state import SupportState, Scenario
from src.environment.actions import parse_model_output, ModelOutput
from src.policies.prompt_templates import (
    SYSTEM_PROMPT_MINIMAL,
    SYSTEM_PROMPT_ENGINEERED,
    build_messages,
)


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON from model output, handling markdown fences."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{[^{}]*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if "```" in pattern else match.group(0)
                return json.loads(candidate)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


class PolicyRunner:
    """Run inference with a HuggingFace model."""

    def __init__(
        self,
        model,
        tokenizer,
        system_prompt: str = SYSTEM_PROMPT_ENGINEERED,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        device: str = "mps",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    def generate(
        self,
        state: SupportState,
        num_return: int = 1,
    ) -> List[str]:
        """Generate model responses for a given state."""
        messages = build_messages(state, self.system_prompt)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=num_return,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the generated tokens (not the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        responses = []
        for output in outputs:
            generated = output[prompt_len:]
            decoded = self.tokenizer.decode(
                generated, skip_special_tokens=True
            )
            responses.append(decoded)

        return responses

    def predict(
        self, state: SupportState
    ) -> Tuple[Optional[dict], str]:
        """Generate one response and parse to dict.
        Returns (parsed_dict_or_None, raw_text)."""
        raw_texts = self.generate(state, num_return=1)
        raw = raw_texts[0]
        parsed = extract_json_from_text(raw)
        return parsed, raw

    def predict_batch(
        self, scenarios: List[Scenario]
    ) -> List[dict]:
        """Run prediction on a batch of scenarios.
        Returns list of {scenario_id, raw, parsed, model_output}."""
        results = []
        for scenario in scenarios:
            parsed, raw = self.predict(scenario.state)
            output = parse_model_output(parsed) if parsed else None
            results.append({
                "scenario_id": scenario.scenario_id,
                "raw_text": raw,
                "parsed": parsed,
                "model_output": (
                    output.to_dict() if output and output.is_valid
                    else None
                ),
                "valid": output is not None and output.is_valid,
            })
        return results
