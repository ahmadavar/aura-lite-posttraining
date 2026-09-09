# POST-TRAINING PROJECT AUDIT

**Date:** 2026-09-07
**Auditor:** Critical technical review (pre-interview)
**Repository:** aura-lite-posttraining/
**Status:** Code complete, NO experiments run yet

---

## PHASE 1 — REPOSITORY FORENSICS

### A. Project Architecture

| Component | Implementation |
|-----------|---------------|
| Base model | Qwen2.5-1.5B-Instruct |
| Model size | ~1.5B parameters |
| Model source | HuggingFace (downloaded to `models/qwen2.5-1.5b-instruct/`) |
| Inference framework | HuggingFace Transformers 5.16.1 |
| Training framework | TRL 1.12.0 (GRPOTrainer / DPOTrainer) + PEFT 0.20.0 (LoRA) |
| Baseline implementation | `scripts/run_baselines.py` — 3 baselines: random, zero-shot, engineered prompt |
| Training implementation | `scripts/train.py` — GRPO or DPO with QLoRA |
| Evaluation implementation | `src/evaluation/metrics.py` + `src/evaluation/evaluate.py` |
| Dataset source | Synthetic, template-based (`scripts/generate_scenarios.py`) |
| Train/Val/Test | 194 / 44 / 76 scenarios (314 total) |
| Reward implementation | `src/rewards/components.py` (7 components) + `src/rewards/reward.py` (compositor) |
| Metric implementation | `src/evaluation/metrics.py` — composite reward, intent acc, action acc, missing info recall, etc. |
| Checkpoint storage | `models/trained/` (empty — no training run yet) |
| Experiment logs | None exist |
| Configuration files | `configs/reward_weights.yaml`, `model_config.yaml`, `training_config.yaml` |
| Tests | 36 tests, 36 passing (`tests/test_environment.py`, `tests/test_reward.py`) |

### B. End-to-End Inference Flow

```
Scenario (state + hidden ground_truth)
  → build_user_prompt(state)  [prompt_templates.py]
  → build_messages(state, system_prompt)  [prompt_templates.py]
  → tokenizer.apply_chat_template()
  → model.generate()
  → extract_json_from_text()  [inference.py]
  → parse_model_output()  [actions.py]
  → RewardCalculator.compute(output, ground_truth)  [reward.py]
  → compute_metrics(results, scenarios, reward_calc)  [metrics.py]
```

### C. Training Flow

**DPO path (default, `--method dpo`):**
```
train scenarios
  → PolicyRunner.generate(state, num_return=4)  [K=4 completions per prompt]
  → score each completion with RewardCalculator.compute()
  → pick best as "chosen", worst as "rejected" (require gap ≥ 0.1)
  → DPOTrainer(model, dataset={prompt, chosen, rejected}, peft_config=LoRA)
  → model weights updated via DPO loss
```

**GRPO path (`--method grpo`):**
```
train prompts (no ground truth passed)
  → GRPOTrainer generates K=4 completions per prompt
  → reward_fn(completions) — SIMPLIFIED heuristic, NOT the real reward
  → GRPO policy gradient update
```

---

## PHASE 2 — TRAINING METHOD CLASSIFICATION

### Critical Finding: TWO DIFFERENT REWARD FUNCTIONS

**This is the single most important finding in the audit.**

The project has **two completely different reward functions**:

1. **`src/rewards/reward.py` (RewardCalculator)** — The compositional 7-component reward function that uses ground truth labels (intent, ideal action, required info, etc.). This is well-designed and tested (22 unit tests).

2. **`scripts/train.py:reward_fn()` (lines 72-126)** — A simplified heuristic used inside GRPO training that does NOT use ground truth. It checks:
   - Format compliance: +0.05 (valid JSON with required fields)
   - Valid action string: +0.15 (action ∈ VALID_ACTIONS)
   - Valid intent string: +0.15 (intent ∈ VALID_INTENTS)
   - Reasoning length > 10 chars: +0.10
   - Missing info list non-empty: +0.10
   - Action detail > 20 chars: +0.10
   - Maximum possible: 0.65

**Why this matters:**

The GRPO `reward_fn` cannot distinguish between a correct and incorrect answer. It rewards ANY valid action, ANY valid intent, ANY non-empty missing info list. A model that always outputs `{"reasoning": "some text here", "intent_assessment": "black_screen", "missing_information": ["anything"], "action": "ASK_CLARIFICATION", "action_detail": "some question that is long enough"}` would score 0.65/0.65 = maximum reward regardless of the actual scenario.

**GRPO as implemented trains format compliance, not task behavior.**

### DPO Path Analysis

The DPO path (`prepare_dpo_dataset`) DOES use the real `RewardCalculator` to score completions and create preference pairs. This is methodologically sound IF:
- The model generates sufficiently diverse completions (temperature=0.8 helps)
- There is meaningful reward variance between completions
- Enough valid preference pairs are generated (requires ≥ 2 valid outputs per scenario with gap ≥ 0.1)

### What parameters are updated?

- **LoRA adapters only**: rank 16, alpha 32, on `q_proj`, `v_proj`, `k_proj`, `o_proj`
- Base model weights are frozen
- This is QLoRA (quantized base + LoRA adapters)

### Classification

| Question | GRPO Path | DPO Path |
|----------|-----------|----------|
| What signal drives updates? | Format-only heuristic | Preference pairs scored by real reward |
| Does it use ground truth? | **NO** | Yes (indirectly, via reward scoring) |
| Environment sampling? | Yes (K=4 completions) | Yes (K=4 completions for pair generation) |
| Policy objective? | GRPO (group relative advantage) | DPO (Bradley-Terry preference) |
| Reward model? | No (rule-based) | No (rule-based preference) |
| Reference model? | Yes (frozen copy for KL) | Yes (frozen copy for KL) |
| Weights change? | Yes (LoRA adapters) | Yes (LoRA adapters) |

### Final Classification

```
ACTUAL METHOD (GRPO path):
  Format-compliance RL. Policy gradient optimization against a reward that
  only measures output structure, not task correctness. Will NOT improve
  customer support decision quality.

ACTUAL METHOD (DPO path):
  Preference optimization using rule-based reward scoring. The reward function
  IS the compositional rubric. This genuinely optimizes for task behavior.
  Technically DPO, not RL — but "offline preference optimization using a
  rule-based reward" is accurate and defensible.

SAFE INTERVIEW LABEL:
  "Preference optimization with a compositional rule-based reward function"
  or "DPO post-training with a multi-component reward rubric"
  If pressed: "The reward function acts as a synthetic preference oracle —
  no human labels needed."

LABELS I SHOULD NOT USE:
  - "Reinforcement learning" (DPO is not RL in the strict sense)
  - "RLHF" (no human feedback)
  - "GRPO post-training" (unless the GRPO reward_fn is fixed)
  - "Reward model" (it's a rule-based rubric, not a learned model)
  - "Fine-tuned on customer data" (all data is synthetic)
```

---

## PHASE 3 — BASELINE AUDIT

### Implemented Baselines

| Property | Random | Zero-shot (Baseline A) | Engineered (Baseline B) | Post-trained |
|----------|--------|----------------------|------------------------|-------------|
| Same model? | N/A | Yes (Qwen2.5-1.5B) | Yes (Qwen2.5-1.5B) | Yes (Qwen2.5-1.5B + LoRA) |
| Same model size? | N/A | Yes | Yes | Yes (base frozen) |
| Same quantization? | N/A | float16 | float16 | float16 |
| Same system prompt? | N/A | MINIMAL | **ENGINEERED** | **ENGINEERED** |
| Same context? | N/A | Yes | Yes | Yes |
| Same max tokens? | N/A | 512 | 512 | 512 |
| Same temperature? | N/A | **0.3** | **0.3** | **0.7** (train) |
| Same test set? | Yes | Yes | Yes | Should be same |
| Same parser? | N/A | Yes | Yes | Yes |
| Same action space? | Yes | Yes | Yes | Yes |
| Same metrics? | Yes | Yes | Yes | Yes |

### IS THE COMPARISON FAIR?

**Partially. Key confounders:**

1. **Prompt difference is massive.** `SYSTEM_PROMPT_MINIMAL` (6 lines, ~50 words) vs `SYSTEM_PROMPT_ENGINEERED` (~300 words with action definitions, rules, intent list, format spec). The engineered prompt contains the ENTIRE intent list (`black_screen, charging, ...`) and action definitions. This prompt engineering alone could account for most improvement.

2. **Temperature mismatch.** Baselines use temperature=0.3 (more deterministic). Training uses temperature=0.7 (more diverse). The post-trained model's inference temperature is not explicitly set in an evaluation script. If the post-trained model is evaluated at a different temperature than baselines, comparison is confounded.

3. **The engineered prompt is the RIGHT baseline.** The post-trained model uses `SYSTEM_PROMPT_ENGINEERED` during training (line 154 in `train.py`). So the fair comparison is:
   - Engineered prompt (no training) vs. Engineered prompt + DPO training
   - This is good experimental design IF temperature is matched.

4. **No evaluation script for the post-trained model exists yet.** `scripts/run_baselines.py` evaluates baselines only. There is no `scripts/evaluate_trained.py`. This needs to be created.

### Verdict on Baselines

The baseline DESIGN is sound (random + weak prompt + strong prompt). The comparison against the strong baseline (Baseline B: engineered prompt) is the one that matters. Beating only the zero-shot baseline would be unconvincing.

---

## PHASE 4 — DATASET AUDIT

### Summary Statistics

| Split | Count | % | Unique Messages |
|-------|-------|---|-----------------|
| Train | 194 | 61.8% | 126 (65%) |
| Val | 44 | 14.0% | 19 (43%) |
| Test | 76 | 24.2% | 44 (58%) |
| **Total** | **314** | 100% | **189** |

### Cross-Split Leakage: NONE

Zero customer message templates appear in more than one split. Template-level splitting was implemented correctly in `split_scenarios()` using device-name stripping and group-level assignment.

### Class Distribution: SEVERELY SKEWED

| Action | Train | Val | Test |
|--------|-------|-----|------|
| ASK_CLARIFICATION | 71% | 77% | **80%** |
| ROUTE_CLAIM | 15% | 7% | **1%** (1 scenario!) |
| SEARCH_KB | 10% | 9% | 14% |
| PROVIDE_STEP | 3% | 7% | 4% |
| ESCALATE | 0% | 0% | 0% |
| COMPLETE | 0% | 0% | 0% |

**Critical problems:**
- **ASK_CLARIFICATION dominates** (71-80%). A model that always outputs ASK_CLARIFICATION gets 71-80% action "accuracy" for free.
- **ROUTE_CLAIM is 15% of train but 1% of test.** The model trains on claim routing but is barely tested on it.
- **ESCALATE and COMPLETE never appear as ideal actions.** Two of the six actions are never the correct answer. They exist only as "wrong answers."
- **Action accuracy is a misleading metric** given this imbalance.

### Scenario Generation Concerns

1. **Template-based generation.** 189 unique templates expanded to 314 via device variants + turn variants. Effective diversity is low.
2. **`issue_category` is always NULL.** Dead field in all 314 scenarios. Not a problem, but dead code.
3. **Ambiguity distribution is designed in** (clear/ambiguous/misleading). This is good for evaluation stratification.
4. **Formulaic structure.** All scenarios follow `"{device} has {problem}. {detail}."` pattern. No multi-issue messages, no angry customers, no irrelevant information.

### Dataset Risk: **MEDIUM**

- No leakage ✓
- Template diversity is adequate for a proof-of-concept
- Class imbalance is a serious problem for evaluation
- Could memorize templates rather than learn behavior (189 unique patterns for 194 train scenarios = nearly 1:1)

---

## PHASE 5 — REWARD FUNCTION AUDIT

See `REWARD_AUDIT.md` for full details. Summary:

### Component Table

| Component | Weight | Purpose | Exploitable? | Risk |
|-----------|--------|---------|--------------|------|
| R_intent (exact match) | 0.20 | Correct intent classification | Low — requires actual understanding or memorization | LOW |
| R_action (ideal=1.0, acceptable=0.5) | 0.25 | Correct next action | **HIGH — always pick ASK_CLARIFICATION for 0.5+** | HIGH |
| R_missing_info (recall - FP penalty) | 0.15 | Identify information gaps | Medium — list common items | MEDIUM |
| R_clarification (keyword match) | 0.15 | Quality of clarification question | **HIGH — keyword stuffing** | HIGH |
| R_efficiency (unnecessary ask penalty) | 0.10 | Don't ask when info exists | Low — only fires for 10% weight | LOW |
| R_should_not (forbidden action) | 0.10 | Avoid forbidden actions | Low — just avoid 1-2 actions | LOW |
| R_format (valid JSON) | 0.05 | Output structure | Low — trivial to satisfy | LOW |

### Primary Exploit Path

A model that always outputs:
```json
{
  "reasoning": "Customer needs help with their device issue",
  "intent_assessment": "black_screen",
  "missing_information": ["physical_damage_check", "recent_events"],
  "action": "ASK_CLARIFICATION",
  "action_detail": "Can you tell me about any physical damage, recent events, or drops?"
}
```

Would score:
- R_intent: 0.0 or 1.0 (10/10 scenarios are black_screen, so 10% hit rate random)
- R_action: 0.5-1.0 (ASK_CLARIFICATION is acceptable/ideal in ~90% of scenarios)
- R_missing_info: variable (common items hit many scenarios)
- R_clarification: high (keywords match many topics)
- R_efficiency: 1.0 (required_information exists in most scenarios)
- R_should_not: 1.0 (ASK_CLARIFICATION is never forbidden)
- R_format: 1.0

**The safest degenerate strategy is "always ask clarification."** This is a real reward hacking risk.

### GRPO reward_fn (train.py) — BROKEN

The GRPO training reward function is a separate, simplified function that does NOT compare against ground truth. It rewards:
- Valid format: +0.05
- Any valid action string: +0.15
- Any valid intent string: +0.15
- Reasoning > 10 chars: +0.10
- Non-empty missing_info: +0.10
- Action detail > 20 chars: +0.10

**This function cannot distinguish correct from incorrect decisions.** It is a format-compliance reward disguised as a task reward. Using this for GRPO training would produce a model that formats output correctly but has no improved decision quality.

---

## PHASE 6 — EVALUATION AUDIT

### Implemented Metrics

| Metric | Implemented? | Location |
|--------|-------------|----------|
| Composite reward (weighted) | Yes | `metrics.py:compute_metrics()` |
| Intent accuracy | Yes | Exact match, case-insensitive |
| Action accuracy (ideal only) | Yes | Exact match to `ideal_action` |
| Action accuracy (acceptable) | **NO** — not tracked separately |
| Missing info recall | Yes | Fuzzy substring matching |
| Missing info precision | **NO** — FP count exists in reward but not in metrics |
| Unnecessary question rate | Yes | ASK when no required_info |
| Format compliance | Yes | Valid outputs / total |
| Per-domain reward | Yes | Breakdown by intent |
| Bootstrap CI | Yes | 1000-sample bootstrap |
| Action distribution | Yes | Counter per action |
| Confusion matrix | **NO** |
| Per-class precision/recall | **NO** |
| Per-action F1 | **NO** |
| Escalation accuracy | **NO** (ESCALATE never ideal) |
| Completion accuracy | **NO** (COMPLETE never ideal) |
| Avg output tokens | Yes | Character count (not tokens) |
| Latency | Partial | `inference_time_sec` for baselines only |

### Missing Metrics That Matter

1. **Acceptable-action accuracy**: The evaluation only counts exact match to `ideal_action`. But 63% of scenarios have multiple acceptable actions. A model that picks an acceptable (not ideal) action gets 0 for action accuracy but 0.5 for R_action reward. This discrepancy between metrics and reward makes interpretation confusing.

2. **Per-class action metrics**: Given 80% ASK_CLARIFICATION in test, overall action accuracy hides per-class performance. Need confusion matrix.

3. **Missing info precision**: The reward penalizes false positives but the evaluation metrics don't track precision separately.

---

## PHASE 7 — BASELINE VS POST-TRAINING COMPARISON

| Metric | Random | Zero-shot | Engineered | Post-Trained (DPO) | Delta vs Engineered |
|--------|--------|-----------|------------|--------------------|--------------------|
| Composite reward | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Intent accuracy | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Action accuracy | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Missing info recall | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Unnecessary question rate | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Format compliance | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Avg output length | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Latency | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

**No experiments have been run. No results exist. There is nothing to compare.**

---

## PHASE 8 — VERDICT

### Grade: **D — No evidence**

**Current implementation does not demonstrate improvement because no experiments have been run.**

However, the project infrastructure ranges from well-designed (reward components, episode logic, data splitting) to critically flawed (GRPO reward function). The path to a convincing result requires:

1. Fixing the GRPO reward function OR committing to DPO only
2. Running baselines
3. Running training
4. Running evaluation
5. Comparing results

### Structural Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Environment design | **B+** | Clean state/action/episode architecture. Ground truth properly hidden. |
| Reward design | **B** for real reward, **F** for GRPO reward | Compositional rubric is well-designed. GRPO reward is broken. |
| Baseline design | **B** | Three baselines with increasing strength. Temperature mismatch is a confounder. |
| Data quality | **C+** | No leakage, but severe class imbalance and low diversity (189 templates). |
| Evaluation | **C** | Core metrics exist but missing per-class analysis, confusion matrix, precision. |
| Training pipeline | **C** | DPO path is sound. GRPO path is broken. No evaluation script for trained model. |
| Tests | **A** | 36/36 passing. Good coverage of reward components and environment. |
| Documentation | **D** | No README, no docs, no results. |
| Reproducibility | **C** | Seeds set, configs exist, but no run scripts or instructions. |

---

## PHASE 19 — REPO CLEANUP FOR INTERVIEW

### Issues Found

1. **No README.md** — Empty repo with no explanation
2. **No docs/ directory** — Specification docs were supposed to be written
3. **`.cache/huggingface/` committed** — Lock files and metadata inside `models/` directory
4. **No `.gitignore`** — Model weights (3GB+), cache files, `__pycache__` would all be committed
5. **`results/` directory doesn't exist** — Scripts assume it
6. **No evaluation script for trained model** — Only baselines have evaluation
7. **Hard-coded paths** — `MODEL_PATH` uses `__file__` relative paths (OK for local, fragile for deployment)
8. **`issue_category` always NULL** — Dead field across all scenarios
9. **`src/training/__init__.py` exists but module is empty** — No training module code
10. **Requirements.txt version bounds too loose** — `torch>=2.0.0` could pull incompatible versions

### No Secrets or Confidential Data

- All data is synthetic
- No API keys
- No company-internal information
- No screenshots or slides committed
- Safe for public repository
