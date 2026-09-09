# REWARD FUNCTION AUDIT

**Date:** 2026-09-07

---

## Overview

The project has **two separate reward implementations**:

1. **Production reward** (`src/rewards/components.py` + `src/rewards/reward.py`) — 7-component compositional rubric using ground truth labels. Well-tested (22 unit tests). Used by DPO training and evaluation.

2. **GRPO training reward** (`scripts/train.py:reward_fn()`, lines 72-126) — Simplified heuristic that does NOT use ground truth. Checks only format and surface features.

---

## Production Reward Components

### Component Details

#### 1. R_intent (weight: 0.20)
- **Implementation:** Exact string match, case-insensitive
- **Scoring:** 1.0 if match, 0.0 if not
- **Exploitable?** LOW. Requires actual understanding or memorization of domain→intent mapping. 10 intents, random baseline gets 10%.
- **Limitation:** No partial credit for semantically close intents (e.g., "screen_damage" vs "damaged_screen")

#### 2. R_action (weight: 0.25)
- **Implementation:** Ideal match = 1.0, acceptable match = 0.5, wrong = 0.0
- **Exploitable?** **HIGH.** ASK_CLARIFICATION is ideal in ~55% and acceptable in ~85% of scenarios. A model that always picks ASK_CLARIFICATION scores:
  - 1.0 × 55% + 0.5 × 30% + 0.0 × 15% = 0.70 expected R_action
  - This is very high for a degenerate strategy
- **The "always ask" strategy is the dominant exploit.**

#### 3. R_missing_info (weight: 0.15)
- **Implementation:** Fuzzy substring recall of required_information. Penalty of -0.1 per false positive.
- **Scoring:** `recall - 0.1 × FP_count`, floored at 0
- **Exploitable?** MEDIUM. Listing every common required_information item (`["physical_damage_check", "recent_events", "cable_tested", "port_condition", "battery_health_percent"]`) would hit many scenarios.
- **Limitation:** Fuzzy matching is one-directional — "damage" matches "physical_damage_check" because `"damage" in "physical_damage_check"` is True. This is very loose.

#### 4. R_clarification (weight: 0.15)
- **Implementation:** Keyword matching against `acceptable_clarification_topics`. Checks topic substrings in question text, then individual words.
- **Scoring:** `hits / len(topics)` or word-overlap heuristic
- **Exploitable?** **HIGH.** A question containing common topic keywords (`"physical damage, recent events, cable type, screen state, insurance"`) would score well across many scenarios via word-level matching.
- **Returns -1.0 sentinel when action ≠ ASK_CLARIFICATION.** Weight is redistributed proportionally to other components.

#### 5. R_efficiency (weight: 0.10)
- **Implementation:** Penalizes ASK_CLARIFICATION when `required_information` is empty
- **Scoring:** 0.0 if asking unnecessarily, 1.0 otherwise
- **Exploitable?** LOW. Only triggers when required_information is empty AND model picks ASK_CLARIFICATION. Few scenarios have empty required_information.
- **This is the only reward component that punishes the "always ask" strategy.** But at 0.10 weight, it's not enough to overcome R_action's 0.25 weight rewarding ASK_CLARIFICATION.

#### 6. R_should_not (weight: 0.10)
- **Implementation:** Check if chosen action is in `should_not_do` list
- **Scoring:** 0.0 if forbidden, 1.0 otherwise
- **Exploitable?** LOW. ASK_CLARIFICATION is never in `should_not_do` for any scenario.

#### 7. R_format (weight: 0.05)
- **Implementation:** `output.is_valid` — checks action ∈ VALID_ACTIONS, missing_info is list, action_detail and reasoning are non-empty
- **Scoring:** 0.0 or 1.0
- **Exploitable?** LOW. Trivial to satisfy.

### Weight Redistribution

When R_clarification returns -1.0 (action ≠ ASK_CLARIFICATION), its 0.15 weight is redistributed proportionally to all other components. This means:
- For non-ASK actions, R_action's effective weight increases from 0.25 to ~0.294
- This makes action correctness even more important for SEARCH_KB/PROVIDE_STEP/ROUTE_CLAIM

---

## Reward Hacking Analysis

### Exploit Path 1: "Always Ask Clarification"

**Strategy:** Always output ASK_CLARIFICATION with keyword-stuffed question.

**Expected scores:**
- R_intent: ~0.10 (random guess among 10 intents)
- R_action: ~0.70 (ideal or acceptable in ~85% of scenarios)
- R_missing_info: ~0.30 (common items hit some scenarios)
- R_clarification: ~0.50 (keyword stuffing)
- R_efficiency: ~0.90 (most scenarios have required_information)
- R_should_not: 1.0 (ASK never forbidden)
- R_format: 1.0

**Weighted total:** ~0.20(0.10) + 0.25(0.70) + 0.15(0.30) + 0.15(0.50) + 0.10(0.90) + 0.10(1.0) + 0.05(1.0) = **0.02 + 0.175 + 0.045 + 0.075 + 0.09 + 0.10 + 0.05 = 0.555**

A model that MEMORIZES intents but still always asks clarification:
- R_intent: ~0.60 (gets most intents right)
- Everything else same
- **Weighted total: ~0.655**

### Exploit Path 2: Memorize Templates

With 194 training scenarios and only 126 unique messages, a model could memorize (message → intent + action) pairs. At test time, similar templates would trigger memorized responses.

**Risk:** MEDIUM. LoRA rank 16 on a 1.5B model has limited capacity for rote memorization, but 126 patterns is not many.

### Exploit Path 3: Keyword Stuffing in Clarification

The R_clarification component uses word-level overlap:
```python
topic_words = set()
for t in topics:
    topic_words.update(t.split("_"))
    topic_words.update(t.split())
```

A question like "Can you describe any physical damage, recent events, cable type, battery health, screen state, water exposure, and insurance status?" would match topic words for nearly every scenario.

**Risk:** HIGH. This is a real exploit that training could discover.

---

## GRPO Reward Function (BROKEN)

### Location: `scripts/train.py`, lines 72-126

### What it checks:
```
+0.05  if JSON parses correctly
+0.15  if action ∈ {"ASK_CLARIFICATION", "SEARCH_KB", ...}
+0.15  if intent ∈ {"black_screen", "charging", ...}
+0.10  if len(reasoning) > 10
+0.10  if missing_information is non-empty list
+0.10  if len(action_detail) > 20
──────
 0.65  maximum possible
```

### What it does NOT check:
- Whether the intent is CORRECT for the scenario
- Whether the action is CORRECT for the scenario
- Whether missing information is RELEVANT
- Whether the clarification question makes sense
- Whether the action violates should_not_do rules

### Verdict: USELESS FOR TASK IMPROVEMENT

This reward function trains the model to:
1. Output valid JSON ✓
2. Pick any valid action string ✓
3. Pick any valid intent string ✓
4. Write enough text ✓
5. Include a list ✓

It does NOT train the model to make better customer support decisions. Using this for GRPO would be equivalent to supervised training on format compliance.

### Recommendation

Either:
- **Fix GRPO reward** to use RewardCalculator (requires passing scenarios alongside prompts)
- **Abandon GRPO, use DPO only** (DPO path already uses the real reward)
- **Rename GRPO to what it is:** format-compliance fine-tuning

---

## Reward Design Quality Assessment

### Strengths
1. Compositional structure allows ablation studies
2. Weight redistribution handles N/A clarification correctly
3. FP penalty in R_missing_info prevents "list everything" strategy
4. R_efficiency provides some penalty for unnecessary asks
5. Configurable via YAML
6. Well-tested (22 unit tests)

### Weaknesses
1. **R_clarification uses keyword matching, not semantic evaluation.** Keyword stuffing is trivially exploitable.
2. **R_efficiency weight (0.10) is too low to counteract R_action (0.25) rewarding ASK_CLARIFICATION.** The "always ask" strategy nets ~0.55-0.65 composite reward.
3. **No penalty for always choosing the same action.** A diversity bonus or action-distribution penalty would help.
4. **R_missing_info fuzzy matching is too loose.** `"damage" in "physical_damage_check"` matches, but so would `"check" in "physical_damage_check"`.
5. **No reward for action_detail quality when action ≠ ASK_CLARIFICATION.** SEARCH_KB detail, PROVIDE_STEP detail, ROUTE_CLAIM detail are not evaluated.
6. **The GRPO reward function is completely broken** and would train the wrong behavior.

### Overall Reward Quality: **B- (production reward) / F (GRPO reward)**
