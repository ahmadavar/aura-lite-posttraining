# EXPERIMENT IMPROVEMENT PLAN

**Date:** 2026-09-07

---

## P0 — MUST FIX BEFORE PRESENTING

### P0-1: Fix or Abandon GRPO Reward Function

**Problem:** `scripts/train.py:reward_fn()` (lines 72-126) does not use ground truth. It rewards any valid format, any valid action string, any valid intent string — regardless of correctness. GRPO training with this reward cannot improve decision quality.

**Why it matters:** If someone reads the GRPO path, they'll see a reward function that doesn't distinguish correct from incorrect answers. This destroys credibility. Even if you only demo DPO, the broken GRPO code is in the repo.

**Proposed change (Option A — Fix GRPO):**
Replace `reward_fn` with a function that accepts `(prompts, completions)` and maps prompts back to scenarios to use RewardCalculator. This requires restructuring how GRPOTrainer receives prompts — the scenarios must be accessible by prompt text.

**Proposed change (Option B — Remove GRPO, commit to DPO):**
Remove GRPO from `train.py`. Rename the script. Document that DPO was chosen because "preference optimization with a rule-based reward oracle is cleaner than online RL for this problem size." This is honest and defensible.

**Expected benefit:** Technical credibility preserved.
**Implementation difficulty:** Option A = MEDIUM (TRL GRPOTrainer reward function signature is tricky). Option B = LOW (delete code, update docs).
**Risk:** Option A might not work with TRL's GRPOTrainer API. Option B loses the "RL" label.
**Files affected:** `scripts/train.py`, `configs/training_config.yaml`, README

### P0-2: Run the Actual Experiments

**Problem:** No baselines run, no training done, no evaluation. The project is code with no results.

**Why it matters:** You cannot present a project with no results. Period.

**Proposed change:**
1. Run `scripts/run_baselines.py` — get baseline numbers
2. Run `scripts/train.py --method dpo` — train the model
3. Create and run `scripts/evaluate_trained.py` — evaluate on test set
4. Save all results to `results/`

**Expected benefit:** Actual numbers to present.
**Implementation difficulty:** LOW (scripts exist, just need to run + create eval script)
**Risk:** Training might fail on MPS, results might show no improvement.
**Files affected:** `results/`, new `scripts/evaluate_trained.py`

### P0-3: Create Evaluation Script for Trained Model

**Problem:** `scripts/run_baselines.py` evaluates baselines only. There is no script to evaluate the post-trained model on the test set.

**Why it matters:** Without this, you can't compare baseline vs post-trained.

**Proposed change:** Create `scripts/evaluate_trained.py` that loads the trained model (base + LoRA adapter), runs inference on the test set with the same parser/metrics, and saves results.

**Expected benefit:** Enables the core comparison.
**Implementation difficulty:** LOW (copy baseline evaluation logic, change model loading to include LoRA)
**Files affected:** New `scripts/evaluate_trained.py`

### P0-4: Match Inference Temperature

**Problem:** Baselines use temperature=0.3. Training uses temperature=0.7-0.8. Post-trained evaluation temperature is undefined.

**Why it matters:** Different temperatures = confounded comparison. Higher temperature = more diverse but potentially less accurate outputs.

**Proposed change:** Use temperature=0.3 for ALL evaluation (baselines AND post-trained). Training temperature can differ.

**Expected benefit:** Fair comparison.
**Implementation difficulty:** LOW
**Files affected:** `scripts/run_baselines.py` (already 0.3), new `scripts/evaluate_trained.py`

### P0-5: Create README.md

**Problem:** No README. Empty repo with no explanation.

**Why it matters:** First thing an interviewer sees. No README = amateur.

**Proposed change:** Create README with: problem statement, approach, setup instructions, how to reproduce, results table, one-paragraph takeaway.

**Expected benefit:** Professional presentation.
**Implementation difficulty:** LOW
**Files affected:** New `README.md`

### P0-6: Create .gitignore

**Problem:** No .gitignore. Model weights (3GB+), __pycache__, .cache directories would all be committed.

**Why it matters:** Pushing 3GB model weights to GitHub is unprofessional and possibly impossible.

**Proposed change:** Add .gitignore excluding models/, __pycache__/, .cache/, *.pyc, results/ (or keep results/ tracked).

**Expected benefit:** Clean repository.
**Implementation difficulty:** TRIVIAL
**Files affected:** New `.gitignore`

---

## P1 — HIGH VALUE

### P1-1: Add Per-Class Action Metrics

**Problem:** Overall action accuracy hides class imbalance. With 80% ASK_CLARIFICATION in test, a model that always asks gets 80% action accuracy.

**Why it matters:** An interviewer who notices the imbalance will ask "isn't your model just always picking ASK_CLARIFICATION?" Without per-class metrics, you can't answer.

**Proposed change:** Add confusion matrix, per-action precision/recall, and macro-F1 to `metrics.py`.

**Expected benefit:** Demonstrates you understand class imbalance and designed evaluation accordingly.
**Implementation difficulty:** LOW
**Files affected:** `src/evaluation/metrics.py`

### P1-2: Add "Acceptable Action" Accuracy Metric

**Problem:** Current action accuracy only counts exact match to `ideal_action`. But 63% of scenarios have multiple acceptable actions. R_action gives 0.5 for acceptable actions, but the metric gives 0.

**Why it matters:** Metrics and reward disagree. A model that consistently picks acceptable (not ideal) actions shows 0% action accuracy but decent reward. Confusing.

**Proposed change:** Add `action_accuracy_acceptable` metric: 1 if action ∈ acceptable_actions, 0 otherwise.

**Expected benefit:** More interpretable evaluation.
**Implementation difficulty:** LOW
**Files affected:** `src/evaluation/metrics.py`

### P1-3: Run Reward Ablation (R_efficiency removed)

**Problem:** No evidence that reward design matters.

**Why it matters:** "I designed a reward function" is a claim. "Removing one component changed model behavior" is evidence. This is the strongest defense of your reward engineering.

**Proposed change:** Train twice: once with full reward, once with R_efficiency weight=0. Compare action distributions — if removing the efficiency penalty causes the model to ask more unnecessary questions, that proves the reward component works.

**Expected benefit:** Strongest possible evidence of reward engineering impact.
**Implementation difficulty:** MEDIUM (requires two training runs)
**Files affected:** `configs/`, results comparison

### P1-4: Collect 5 Behavioral Change Examples

**Problem:** Aggregate metrics don't tell a story. "Reward went from 0.45 to 0.52" means nothing to an interviewer.

**Why it matters:** Concrete examples are 10x more convincing than numbers. "On this scenario, the baseline always asked for more info, but the post-trained model recognized it had enough and searched the knowledge base" is a compelling story.

**Proposed change:** After experiments, manually inspect test results. Find 3-5 cases where:
- Baseline chose wrong action, post-trained chose right
- Explain why in plain English
- Include the actual model outputs

**Expected benefit:** Interview-ready behavioral evidence.
**Implementation difficulty:** LOW (manual inspection after experiments)
**Files affected:** New `docs/BEHAVIORAL_EXAMPLES.md`

### P1-5: Add Stratified Evaluation by Ambiguity Level

**Problem:** Scenarios have clear/ambiguous/misleading labels in the generation code, but this is not tracked in the scenario data or evaluation metrics.

**Why it matters:** "Post-training improved accuracy on ambiguous cases by 15%" is much more interesting than "overall accuracy improved by 5%."

**Proposed change:** Add ambiguity level to scenario metadata. Compute metrics per ambiguity level.

**Expected benefit:** Richer analysis, better storytelling.
**Implementation difficulty:** MEDIUM (need to regenerate data with ambiguity labels, update metrics)
**Files affected:** `scripts/generate_scenarios.py`, `src/evaluation/metrics.py`, regenerate data

---

## P2 — NICE TO HAVE

### P2-1: Multiple Random Seeds

**Problem:** Single seed for training. Results might be lucky/unlucky.

**Why it matters:** Confidence intervals on a single run are bootstrap CIs over test scenarios, not over training randomness.

**Proposed change:** Run training with 3 different seeds. Report mean ± std across seeds.

**Expected benefit:** Stronger statistical claims.
**Implementation difficulty:** HIGH (3x training time)
**Files affected:** Training scripts, results

### P2-2: Data Size Ablation

**Problem:** No evidence that more data helps.

**Why it matters:** "Performance scales with training data" is a sign of real learning. "Performance is flat regardless of data" suggests memorization or ceiling effects.

**Proposed change:** Train on 25%, 50%, 100% of training data. Plot learning curve.

**Expected benefit:** Evidence of scaling behavior.
**Implementation difficulty:** MEDIUM (3 training runs)
**Files affected:** Training scripts, results

### P2-3: Improve R_clarification with Sentence Similarity

**Problem:** R_clarification uses keyword matching, which is easily exploitable.

**Why it matters:** A more semantic evaluation would be more robust to keyword stuffing and more convincing to reviewers.

**Proposed change:** Replace keyword matching with sentence-transformer cosine similarity between clarification question and topic descriptions.

**Expected benefit:** More robust evaluation, harder to game.
**Implementation difficulty:** MEDIUM (add sentence-transformers dependency, update component)
**Files affected:** `src/rewards/components.py`, `requirements.txt`

### P2-4: Add Novel Test Scenarios

**Problem:** All test scenarios come from the same template generator. No truly novel inputs.

**Why it matters:** Cannot distinguish memorization from generalization.

**Proposed change:** Manually write 15-20 novel customer messages (not from templates). Include multi-issue, angry, contradictory, already-tried-steps cases.

**Expected benefit:** Evidence of generalization (or honest acknowledgment of its absence).
**Implementation difficulty:** MEDIUM (manual scenario creation + ground truth labeling)
**Files affected:** New `data/processed/test_novel.json`, evaluation scripts

### P2-5: Streamlit Demo Dashboard

**Problem:** No visual demo.

**Why it matters:** "Let me show you" is more powerful than "let me tell you."

**Proposed change:** Simple Streamlit app: input a customer message → show baseline output vs post-trained output side by side with reward breakdown.

**Expected benefit:** Live demo for interviews.
**Implementation difficulty:** MEDIUM
**Files affected:** New `app.py`

---

## ABLATION STUDIES (ranked by value)

| Ablation | Question | Value | Time |
|----------|----------|-------|------|
| Full reward vs no R_efficiency | Does reward design change behavior? | **HIGH** | 2x training |
| Simple vs engineered vs post-trained | Does training beat strong prompt? | **HIGH** | Already planned |
| 25% / 50% / 100% data | Does performance scale? | **MEDIUM** | 3x training |
| Individual reward components | Which signal drives improvement? | **MEDIUM** | 7x training |
| Template vs novel test scenarios | Memorization or generalization? | **HIGH** | Manual effort |

---

## IDEAL FINAL EXPERIMENT DESIGN

### Model
Qwen2.5-1.5B-Instruct (same for all conditions)

### Conditions
| Condition | Model | Prompt | Training | Temperature |
|-----------|-------|--------|----------|-------------|
| A (Random) | N/A | N/A | None | N/A |
| B (Zero-shot) | Qwen2.5-1.5B | MINIMAL | None | 0.3 |
| C (Engineered) | Qwen2.5-1.5B | ENGINEERED | None | 0.3 |
| D (Post-trained) | Qwen2.5-1.5B + LoRA | ENGINEERED | DPO, 3 epochs | 0.3 |
| E (Ablation: no efficiency) | Qwen2.5-1.5B + LoRA | ENGINEERED | DPO, 3 epochs, R_eff=0 | 0.3 |

### Data
- Train: 194 scenarios (DPO preference pairs generated from these)
- Val: 44 scenarios (early stopping / hyperparameter selection)
- Test: 76 scenarios (frozen, never seen during training)
- Novel test (optional): 15-20 hand-written scenarios

### Evaluation (same for all conditions)
- Same test set
- Same parser
- Same action space
- Same metrics
- Same reward weights

### Metrics
1. Composite reward (mean ± bootstrap 95% CI)
2. Intent accuracy
3. Action accuracy (ideal) + Action accuracy (acceptable)
4. Per-action precision/recall (confusion matrix)
5. Missing info recall
6. Missing info precision (FP rate)
7. Unnecessary question rate
8. Format compliance
9. Action distribution (histogram)
10. Per-domain reward breakdown
11. 5 concrete behavioral change examples

### Report
- Aggregate comparison table
- Per-action performance breakdown
- Failure category analysis
- Reward ablation comparison (D vs E)
- 3-5 behavioral examples with commentary
- One-paragraph conclusion
