# Synthetic Data Specification

## 1. Overview

AURA-Lite uses synthetic customer support scenarios for RL post-training of a language model that triages mobile device issues. Each scenario represents a single customer message paired with ground truth annotations describing the ideal agent action, acceptable alternatives, and actions to avoid.

The data is synthetic for three reasons:

1. **No access to real Asurion support transcripts.** Proprietary customer data is unavailable for academic practicum work.
2. **Controlled action distribution.** Real support logs are heavily skewed toward a few action types (mostly clarification and troubleshooting steps). Synthetic generation lets us target a balanced distribution so the reward model does not collapse onto a single dominant action.
3. **Ground truth labeling at scale.** Each scenario includes a structured `hidden_ground_truth` block with ideal action, acceptable actions, required information, and prohibited actions. Producing this for hundreds of real conversations would be prohibitively expensive.

The dataset covers 10 device-issue domains (black_screen, charging, battery_drain, activation, wifi_network, data_transfer, damaged_screen, lost_stolen, liquid_damage, claim_routing) across 13 device models and 6 discrete action types.

---

## 2. Schema

Each scenario is a JSON object with the following structure:

```json
{
  "scenario_id": "scn_0001",
  "ambiguity_level": "ambiguous | clear | misleading",
  "difficulty": "easy | medium | hard",
  "template_family": "black_screen_ambiguous_ask",
  "state": {
    "customer_message": "My phone isn't working. Screen is dark I think.",
    "device_type": "iPhone 15" or null,
    "issue_category": null,
    "information_collected": [],
    "conversation_turn": 0,
    "available_kb_topics": ["force_restart_iphone", "screen_replacement_claim", ...]
  },
  "hidden_ground_truth": {
    "true_intent": "black_screen",
    "true_issue": "software_crash",
    "required_information": ["physical_damage_check", "recent_events", ...],
    "ideal_action": "ASK_CLARIFICATION",
    "ideal_clarification": "Clarification for black_screen issue",
    "acceptable_actions": ["ASK_CLARIFICATION"],
    "acceptable_clarification_topics": ["physical_damage", "screen_state", "recent_events"],
    "resolution_path": "force_restart",
    "should_not_do": ["COMPLETE", "ROUTE_CLAIM", "ESCALATE"]
  }
}
```

### Field definitions

**Top-level metadata:**
| Field | Type | Description |
|-------|------|-------------|
| `scenario_id` | string | Unique ID, format `scn_NNNN` for generated or `challenge_NNN` for challenge set |
| `ambiguity_level` | enum | `ambiguous` (missing key details), `clear` (enough info to act), `misleading` (customer's self-diagnosis is wrong) |
| `difficulty` | enum | `easy`, `medium`, `hard` -- reflects how tricky the correct action choice is |
| `template_family` | string | Groups scenarios from the same generation template; used for leakage-free splitting |

**State (observable by the model):**
| Field | Type | Description |
|-------|------|-------------|
| `customer_message` | string | The customer's verbatim message |
| `device_type` | string or null | Device model; null ~30% of the time to simulate missing info |
| `issue_category` | string or null | Always null at turn 0; populated in multi-turn variants |
| `information_collected` | list[string] | Facts already gathered in prior turns |
| `conversation_turn` | int | 0 for first contact; 1-2 for follow-ups (COMPLETE and boosted scenarios) |
| `available_kb_topics` | list[string] | 4 randomly sampled KB article slugs from the scenario's domain |

**Ground truth (hidden, used only by reward function):**
| Field | Type | Description |
|-------|------|-------------|
| `true_intent` | string | One of 10 domain categories |
| `true_issue` | string | Specific sub-issue, e.g. `software_crash`, `cracked_screen` |
| `required_information` | list[string] | Info still needed before acting; empty means enough info exists |
| `ideal_action` | string | Single best action from the 6-action space |
| `ideal_clarification` | string | Placeholder description of what to clarify |
| `acceptable_actions` | list[string] | All actions that should receive positive reward |
| `acceptable_clarification_topics` | list[string] | Valid topics if clarifying |
| `resolution_path` | string | Expected downstream resolution, e.g. `force_restart`, `theft_claim` |
| `should_not_do` | list[string] | Actions that should receive negative reward |

---

## 3. Generation Pipeline

The script `scripts/generate_scenarios.py` produces the dataset in four phases with a fixed seed (`SEED = 42`).

### Phase 1: Domain templates (bulk of data)

36 domain-specific templates are defined across the 10 domains in the `TEMPLATES` dict. Each template contains:
- A list of 2-5 message variants (paraphrases of the same scenario)
- Metadata: ambiguity level, difficulty, template family name
- Full ground truth block

For each message variant, 2-3 devices are randomly sampled from a pool of 13 devices. Messages with 4+ variants get 2 device samples; those with fewer get 3. The `{device}` placeholder is replaced at generation time.

`device_type` in the state is set to null with 30% probability (`random.random() > 0.3`) to simulate customers who do not mention their device model.

### Phase 2: Escalation templates

5 cross-domain escalation templates cover safety hazards (swelling battery), repeated failed troubleshooting, multi-system failures, bad prior agent experience, and chronic unresolved issues. Each message gets 2 device samples.

### Phase 3: Completion templates

6 cross-domain completion templates cover resolved issues, confirmed fixes, simple info requests, and completed claims. Each message gets 3 device samples. These scenarios have `conversation_turn` set to 1 or 2 and `information_collected` populated with 1-3 randomly sampled prior-turn facts.

### Phase 4: Boosting pass

30 ASK_CLARIFICATION scenarios are cloned and modified to represent a later conversation turn (turn 2) where most required information has already been collected. The ideal action is shifted:
- Domains `claim_routing`, `damaged_screen`, `lost_stolen`, `liquid_damage` shift to ROUTE_CLAIM
- Other domains shift to either SEARCH_KB or PROVIDE_STEP (randomly chosen)

The boosted variant's `template_family` gets a `_boosted` suffix to keep it identifiable.

After all four phases, the full scenario list is shuffled.

### Output

The script writes 4 files:
- `data/raw/scenarios.json` -- all 392 scenarios
- `data/processed/train.json` -- 266 scenarios
- `data/processed/val.json` -- 43 scenarios
- `data/processed/test.json` -- 83 scenarios

---

## 4. Action Distribution

### Target distribution (from script docstring)

| Action | Target % |
|--------|----------|
| ASK_CLARIFICATION | 25-30% |
| SEARCH_KB | 20% |
| PROVIDE_STEP | 20% |
| ROUTE_CLAIM | 15% |
| ESCALATE | 5-8% |
| COMPLETE | 5-8% |

### Actual distribution (n=392)

| Action | Count | % |
|--------|-------|---|
| ASK_CLARIFICATION | 105 | 26.8% |
| SEARCH_KB | 88 | 22.4% |
| PROVIDE_STEP | 82 | 20.9% |
| ROUTE_CLAIM | 49 | 12.5% |
| ESCALATE | 32 | 8.2% |
| COMPLETE | 36 | 9.2% |

### Why rebalanced

A naive generation approach (one template per domain, each domain naturally asks for clarification first) would produce ~60%+ ASK_CLARIFICATION scenarios. This causes two problems for RL training:

1. **Reward model collapse.** If the model learns that ASK_CLARIFICATION is almost always correct, it never learns the policy boundary between asking and acting. The reward signal becomes trivially exploitable.
2. **Poor downstream behavior.** A model that always asks for clarification is safe but unhelpful. Real support interactions require the agent to act (search KB, provide a step, route a claim) when enough information is present.

The rebalancing ensures every action type appears at meaningful frequency, so the reward model must learn context-dependent decision boundaries. The Phase 4 boosting pass specifically addresses the ask-vs-act boundary by showing the same scenario at different conversation turns with different ideal actions.

---

## 5. Split Strategy

### Sizes

| Split | Count | % of total |
|-------|-------|------------|
| Train | 266 | 67.9% |
| Val | 43 | 11.0% |
| Test | 83 | 21.2% |

### Template-level split (no leakage)

Splitting is done at the **template_family** level, not the individual scenario level. All scenarios from the same template family go into the same split. This prevents near-duplicate leakage: message variants within a family share the same underlying structure, ground truth, and phrasing patterns.

The split procedure:
1. Group all scenarios by `template_family` (47 base families + 12 boosted = 59 total families).
2. Group template families by `ideal_action` to ensure stratification.
3. For each action type:
   - If 2 or fewer families: all go to train (cannot split without losing the action in a split).
   - If 3-4 families: 1 to val, 1 to test, rest to train.
   - If 5+ families: 70% to train, 15% to val, remainder to test (with at least 1 group guaranteed in each).
4. Each split is shuffled independently.

Leakage is verified by `check_leakage()`, which confirms zero intersection of template families across splits.

### Train split action distribution (n=266)

| Action | Count | % |
|--------|-------|---|
| ASK_CLARIFICATION | 73 | 27.4% |
| SEARCH_KB | 61 | 22.9% |
| PROVIDE_STEP | 52 | 19.5% |
| ROUTE_CLAIM | 38 | 14.3% |
| ESCALATE | 18 | 6.8% |
| COMPLETE | 24 | 9.0% |

### Challenge set

A separate hand-crafted set of 26 scenarios lives at `data/challenge/scenarios.json`. These do not share template families with the main dataset (they have no `template_family` field; they use `challenge_category` instead). The challenge set tests generalization across 10 categories:

| Category | Count | Description |
|----------|-------|-------------|
| paraphrased | 5 | Same issue, completely different wording |
| multi_issue | 4 | Customer mentions multiple problems |
| angry_frustrated | 3 | Emotional/hostile tone |
| already_tried | 3 | Customer lists prior troubleshooting |
| contradictory | 3 | Customer gives conflicting information |
| enough_info | 4 | All info present, model should act not ask |
| edge_case_unknown_device | 1 | Unrecognized device model |
| edge_case_short_message | 1 | Extremely terse message |
| edge_case_unrelated | 1 | Off-topic request |
| edge_case_garbled | 1 | Incoherent/garbled text |

Challenge action distribution (n=26): ASK_CLARIFICATION 11, SEARCH_KB 4, ROUTE_CLAIM 4, PROVIDE_STEP 3, ESCALATE 3, COMPLETE 1.

---

## 6. Ground Truth Design

Each scenario's `hidden_ground_truth` encodes a multi-dimensional judgment, not just a label.

### ideal_action

The single best action for the scenario. Used as the primary reward signal. One of 6 values: ASK_CLARIFICATION, SEARCH_KB, PROVIDE_STEP, ROUTE_CLAIM, ESCALATE, COMPLETE.

### acceptable_actions

A list of 1-3 actions that should receive positive (though possibly reduced) reward. The ideal action is always in this list. For example, a clear black screen scenario has `ideal_action: SEARCH_KB` and `acceptable_actions: [SEARCH_KB, PROVIDE_STEP]` -- searching the knowledge base is best, but providing a troubleshooting step directly is also reasonable.

### missing_info (required_information)

A list of information items that the agent still needs before it can resolve the issue. When this list is non-empty, ASK_CLARIFICATION is typically the ideal action. When empty, the scenario has enough context for the agent to act.

This field is central to the ask-vs-act boundary. The Phase 4 boosted scenarios demonstrate this: the same template starts with `required_information: [item1, item2, item3]` at turn 0 (ideal = ASK), then at turn 2 with most items in `information_collected`, the ideal shifts to SEARCH_KB or PROVIDE_STEP.

### ideal_intent (true_intent + true_issue)

Two fields capture the ground truth diagnosis:
- `true_intent`: one of 10 domain categories (e.g., `black_screen`, `charging`)
- `true_issue`: specific sub-issue within the domain (e.g., `software_crash`, `charging_port_debris`, `battery_end_of_life`)

These are used by the reward function to evaluate the model's `intent_assessment` output.

### should_not_do

A list of actions that should receive negative reward. For example, a scenario where the customer reports a resolved issue has `should_not_do: [ASK_CLARIFICATION, SEARCH_KB, PROVIDE_STEP]` -- continuing to troubleshoot a resolved issue is actively harmful.

### acceptable_clarification_topics

When the ideal action is ASK_CLARIFICATION, this list specifies which topics the clarifying question should address (e.g., `physical_damage`, `screen_state`). Used by the reward function to evaluate whether the model asks about the right thing, not just whether it asks at all.

---

## 7. Limitations

### Synthetic vs. real data

1. **Limited linguistic diversity.** Each template has 2-5 message variants written by a single author. Real customers use vastly more diverse vocabulary, grammar, and tone. The 13-device pool and `{device}` substitution adds some surface variation, but the underlying sentence structure is fixed per template.

2. **No multi-turn dynamics.** Most scenarios represent a single customer turn (turn 0). The COMPLETE and boosted scenarios simulate later turns by setting `conversation_turn > 0` and pre-populating `information_collected`, but there is no actual conversation history. Real support interactions involve back-and-forth dialogue where earlier responses shape later messages.

3. **Simplified ambiguity.** The three-level ambiguity classification (ambiguous/clear/misleading) is a rough approximation. Real customer messages exist on a continuous spectrum of clarity and often combine clear and ambiguous elements within a single message.

4. **No noise or typos.** All messages are grammatically correct and coherent (except 1 garbled edge case in the challenge set). Real support messages frequently contain typos, autocorrect errors, incomplete sentences, and code-switching between languages.

5. **Uniform customer profiles.** The data has no variation in customer history, account status, warranty terms, or prior interaction count. Real triage decisions depend heavily on these contextual factors.

### Template diversity

- 47 base template families across 10 domains (average 4.7 per domain, range 3-4 per domain in the main TEMPLATES dict, plus escalation and completion cross-domain templates).
- 59 total families including 12 boosted variants.
- Within each family, message variants are close paraphrases sharing the same sentence structure and key phrases. This means models can achieve high accuracy by learning template-level patterns rather than generalizable triage reasoning.
- The challenge set (26 scenarios, no template family overlap) partially mitigates this by testing with novel phrasings, but 26 samples is too small for robust evaluation of generalization.
- The `misleading` ambiguity level is underrepresented at 15 scenarios (3.8% of total). This category is arguably the most important for RL training (teaching the model not to trust the customer's self-diagnosis), but it has the fewest examples.
