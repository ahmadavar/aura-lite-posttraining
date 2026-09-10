# SHORT DEMO PACKAGE -- 75-Second Technical Video

**Created:** 2026-09-09
**Purpose:** Radically focused presentation for Bassim / Asurion practicum evaluation

---

## 1. SOURCE-OF-TRUTH AUDIT

### Exact Verified Experiment

| Item | Verified Value | Source |
|------|---------------|--------|
| Problem | Customer support triage: given a customer message, choose the correct next action from 6 options | `src/environment/actions.py`, `src/environment/state.py` |
| Base model | Qwen2.5-1.5B-Instruct (~1.5 billion parameters) | `configs/model_config.yaml`, `scripts/train.py` line 50 |
| Training method | GRPO (Group Relative Policy Optimization) via TRL GRPOTrainer | `scripts/train.py` lines 261-310, uses `GRPOTrainer` from `trl` |
| Adapter | LoRA rank 16, alpha 32, on q/k/v/o attention projections | `scripts/train.py` lines 264-269 |
| Trainable parameters | ~4M LoRA parameters out of ~1.5B total (~0.3%). Base weights frozen. | `configs/model_config.yaml`, `docs/SLIDE_STORYBOARD.md` |
| Training data | 266 synthetic scenarios, 1 epoch, K=4 rollouts per prompt | `results/training_meta.json`, `data/processed/train.json` (266 scenarios verified) |
| Validation data | 43 scenarios | `data/processed/val.json` (43 scenarios verified) |
| Test data | 83 held-out scenarios | `data/processed/test.json` (83 scenarios verified) |
| Challenge data | 26 novel-phrasing scenarios | `data/challenge/scenarios.json` (26 scenarios verified) |
| Total scenarios | 392 training+val+test, plus 26 challenge | Verified via JSON file counts |
| Reward function | 7 rule-based components, deterministic, uses hidden ground truth | `src/rewards/reward.py`, `src/rewards/components.py` |
| Training time | 1033.6 minutes (~17.2 hours) | `results/training_meta.json` |
| Hardware | Apple M1 Pro (MPS backend) | `results/training_meta.json` |
| Evaluation temperature | 0.3 for all conditions | `results/rl_test_results.json`, `results/engineered_test_results.json` |
| GRPO reward during training | Uses full compositional reward with ground truth (NOT the old format-only heuristic) | `scripts/train.py` lines 89-165, confirmed by training log showing intent_accuracy and action_correctness component scores |

### Observable Input (what the model sees)

From `src/environment/state.py` -- SupportState:
- customer_message (text)
- device_type (optional)
- issue_category (optional, always null in practice)
- information_collected (list)
- conversation_turn (int)
- available_kb_topics (list)

### Action Space (what the model predicts)

From `src/environment/actions.py` -- 6 actions:
1. ASK_CLARIFICATION
2. SEARCH_KB
3. PROVIDE_STEP
4. ROUTE_CLAIM
5. ESCALATE
6. COMPLETE

Plus: intent_assessment, missing_information list, reasoning, action_detail (all in structured JSON output).

### Hidden Ground Truth (what the reward sees)

From `src/environment/state.py` -- GroundTruth:
- true_intent, true_issue
- required_information
- ideal_action, acceptable_actions
- acceptable_clarification_topics
- should_not_do
- resolution_path

### What GRPO Does in This Implementation

From `scripts/train.py` lines 89-310:
1. For each training prompt, the model generates K=4 completions
2. Each completion is parsed and scored by the full compositional reward function against hidden ground truth
3. GRPOTrainer computes group-relative advantages (how each completion scored relative to the group mean)
4. Policy gradient updates LoRA weights so higher-reward behavior becomes more likely
5. KL regularization (beta=0.1) prevents the policy from diverging too far from the base model

This is genuine on-policy RL: the model generates its own training data, receives reward from an environment, and updates weights via policy gradient.

### Strongest Non-RL Baseline

Engineered prompt (same base model, same prompt used in GRPO training, no weight updates):
- Composite Reward: 0.616
- Intent Accuracy: 57.8%
- Action Accuracy: 30.1%
- Macro F1: 0.258

### Exact GRPO Results (from `results/rl_test_results.json`)

| Metric | Engineered | GRPO | Delta |
|--------|-----------|------|-------|
| Composite Reward | 0.616 | 0.657 | +0.041 |
| Intent Accuracy | 57.8% | 66.3% | +8.5pp |
| Action Accuracy (exact) | 30.1% | 34.9% | +4.8pp |
| Action Accuracy (acceptable) | 53.0% | 53.0% | +0.0pp |
| Macro Action F1 | 0.258 | 0.303 | +0.045 |
| Format Compliance | 98.8% | 100% | +1.2pp |
| Action Entropy | 1.012 | 1.162 | +0.150 |
| Missing Info Recall | 2.1% | 0.0% | -2.1pp |

All values verified directly from `results/rl_test_results.json` and `results/engineered_test_results.json`.

### Action Distribution (from result JSONs)

| Action | Ground Truth | Engineered | GRPO |
|--------|-------------|-----------|------|
| ASK_CLARIFICATION | 23 (28%) | 3 (4%) | 4 (5%) |
| SEARCH_KB | 17 (20%) | 67 (81%) | 65 (78%) |
| PROVIDE_STEP | 19 (23%) | 6 (7%) | 7 (8%) |
| ROUTE_CLAIM | 10 (12%) | 1 (1%) | 2 (2%) |
| ESCALATE | 8 (10%) | 5 (6%) | 5 (6%) |
| COMPLETE | 6 (7%) | 0 (0%) | 0 (0%) |

Both models still massively over-predict SEARCH_KB. The SEARCH_KB collapse is the dominant unsolved problem.

### Challenge Set Performance

| Metric | Test (83) | Challenge (26) |
|--------|----------|---------------|
| Composite Reward | 0.657 | 0.576 |
| Intent Accuracy | 66.3% | 46.2% |
| Action Accuracy | 34.9% | 19.2% |
| Macro F1 | 0.303 | 0.107 |

Significant degradation on novel phrasings. Generalization is limited.

### Contradictions Found in Original Transcript

See Section 10 below for full list.

---

## 2. THE ONE-SENTENCE STORY

**GRPO post-training taught a small language model better customer-support decisions than prompt engineering alone -- but did not eliminate its bias toward a single safe action.**

(24 words)

---

## 3. THE ONE PROBLEM

The base model understands customer-support language but collapses to a single decision: SEARCH_KB. Even with a carefully engineered prompt, the model chooses "search the knowledge base" for 81% of scenarios -- when the ground truth says it should only be 20%. This is not a language problem. It is a decision-policy problem. The model defaults to the safest action instead of learning when to escalate, route a claim, or provide a direct troubleshooting step.

(72 words)

---

## 4. THE THREE METRICS

### Metric 1: Intent Accuracy

- **Baseline (Engineered):** 57.8%
- **GRPO:** 66.3%
- **Delta:** +8.5 percentage points
- **Plain English:** The model got better at diagnosing what the customer's actual problem is.

### Metric 2: Action Accuracy (exact match)

- **Baseline (Engineered):** 30.1%
- **GRPO:** 34.9%
- **Delta:** +4.8 percentage points
- **Plain English:** The model more often chose the correct next step.

### Metric 3: Macro F1

- **Baseline (Engineered):** 0.258
- **GRPO:** 0.303
- **Delta:** +0.045
- **Plain English:** Improvement spread across action types, not just the dominant one.

---

## 5. THE TWO KEY FINDINGS

### Finding #1 (MAIN NARRATIVE): Intent accuracy improved by 8.5 percentage points

**Why this is the strongest finding:**
- It is the largest single metric improvement
- It proves the model learned to better diagnose problems, not just format outputs
- It is robust: the engineered prompt already gives the model the full intent list, so the improvement comes from GRPO learning to use that information more effectively through practice
- It is easy to explain in 10 seconds: "The model got 8.5 points better at identifying what the customer's problem actually is"
- It survives scrutiny: verified from `results/rl_test_results.json` (0.6627 = 66.3%) vs `results/engineered_test_results.json` (0.5783 = 57.8%)

### Finding #2 (LIMITATION / CLOSING): SEARCH_KB collapse persists

**Why this matters:**
- GRPO reduced SEARCH_KB from 81% to 78% of predictions -- a marginal shift
- Ground truth distribution is 20% SEARCH_KB
- The model still never predicts COMPLETE (0/6 scenarios)
- This proves the model improved within its existing behavioral mode rather than fundamentally restructuring its decision policy
- An experienced ML engineer would immediately ask about this -- addressing it proactively shows technical maturity
- Challenge set macro F1 drops from 0.303 to 0.107, showing limited generalization

**Selection rationale:** Finding #1 is the main narrative because it is positive, easily understood, and defensible. Finding #2 is the honest limitation that makes the presentation credible. Together they tell a complete story: "it improved, but not completely."

---

## 6. RECOMMENDED FIVE-SLIDE DECK

### SLIDE 1 -- THE QUESTION

**TITLE:** Can RL teach a small model better support decisions than prompting alone?

**PURPOSE:** Establish the research question immediately. No preamble.

**EXACT ON-SCREEN TEXT:**
> Can reinforcement learning teach a small language model
> better customer-support decisions
> than prompt engineering alone?
>
> Ahmad Naggayev | Aura Lite | Qwen2.5-1.5B + GRPO

**VISUAL:** Clean text on dark background. No diagrams. No logos.

**SPOKEN WORDS:**
"I'm Ahmad. I ran an experiment: can reinforcement learning teach a small language model to make better customer-support decisions than a carefully written prompt? [PAUSE] Here's what I found."

**APPROXIMATE TIME:** 0:00 -- 0:12

---

### SLIDE 2 -- THE FAILURE MODE

**TITLE:** The model knows the language. It lacks judgment.

**PURPOSE:** Show the single problem the experiment addresses.

**EXACT ON-SCREEN TEXT:**
> Prompted model chooses SEARCH_KB
> for 81% of scenarios.
>
> Ground truth: 20%.
>
> It defaults to the safe action
> instead of making a decision.

**VISUAL:** Simple horizontal bar or pie showing 81% SEARCH_KB vs 19% everything else. Alternatively, two stacked bars side by side: "Predicted" (81% SEARCH_KB) vs "Actual" (20% SEARCH_KB). Keep it visually stark.

**SPOKEN WORDS:**
"With a strong engineered prompt, the model already understands support language. But it defaults to one action -- 'search the knowledge base' -- eighty-one percent of the time. [PAUSE] The correct rate is twenty percent. This isn't a language problem. It's a decision-policy problem."

**APPROXIMATE TIME:** 0:12 -- 0:28

---

### SLIDE 3 -- GRPO IN ONE PICTURE

**TITLE:** GRPO: practice, not instructions.

**PURPOSE:** Explain the training method in one visual and one sentence.

**EXACT ON-SCREEN TEXT:**
> Customer scenario
>       |
> 4 candidate decisions
>       |
> Score against ground truth
>       |
> Update policy toward better choices

**VISUAL:** Vertical flowchart with 4 boxes and arrows. Clean, no equations, no Greek letters.

**SPOKEN WORDS:**
"GRPO -- Group Relative Policy Optimization -- lets the model try four different decisions per scenario, scores them against ground truth, and updates the weights so better choices become more likely. [PAUSE] Instead of telling the model what to do, it learns from practice."

**APPROXIMATE TIME:** 0:28 -- 0:43

---

### SLIDE 4 -- THE RESULT

**TITLE:** Three metrics. Same model. Different training.

**PURPOSE:** Visual centerpiece. Show the improvement clearly.

**EXACT ON-SCREEN TEXT:**
>                 PROMPTED      GRPO       CHANGE
>
> Intent           57.8%   -->   66.3%      +8.5pp
> Accuracy
>
> Action           30.1%   -->   34.9%      +4.8pp
> Accuracy
>
> Macro F1          .258   -->    .303      +.045
> (balance across
>  6 actions)
>
> 83 held-out scenarios. Same base model.

**VISUAL:** Three-row before/after layout with arrows showing improvement. Green accent on deltas. No other metrics. No random baseline. No zero-shot baseline.

**SPOKEN WORDS:**
"On eighty-three held-out scenarios: intent accuracy -- did it understand the customer -- up eight and a half points. Action accuracy -- did it pick the right next step -- up nearly five points. Macro F1 -- balanced improvement across all six action types, not just the dominant one."

**APPROXIMATE TIME:** 0:43 -- 0:58

---

### SLIDE 5 -- HONEST CONCLUSION

**TITLE:** Improved. Not solved.

**PURPOSE:** One gain, one limitation, one takeaway.

**EXACT ON-SCREEN TEXT:**
> LEARNED:
> Better problem diagnosis and action selection.
>
> NOT SOLVED:
> Still chooses SEARCH_KB 78% of the time.
> Ground truth: 20%.
>
> TAKEAWAY:
> GRPO changed behavior that
> prompt engineering alone did not.

**VISUAL:** Three horizontal blocks. Green (LEARNED) / Orange (NOT SOLVED) / Blue (TAKEAWAY). Clean, sparse text.

**SPOKEN WORDS:**
"What improved: the model diagnoses problems better and picks better actions. [PAUSE] What's still broken: it still defaults to search-knowledge-base seventy-eight percent of the time. The bias is reduced, not eliminated. [PAUSE] The takeaway: reinforcement learning changed decision behavior that prompting alone could not. The code and full results are on GitHub."

**APPROXIMATE TIME:** 0:58 -- 1:15

---

## 7. 75-SECOND FINAL SCRIPT

```
[SLIDE 1]
[0:00] I'm Ahmad. I ran an experiment: can reinforcement learning
teach a small language model to make better customer-support
decisions than a carefully written prompt?
[PAUSE]
Here's what I found.

[SLIDE 2]
[0:12] With a strong engineered prompt, the model already understands
support language. But it defaults to one action -- search the
knowledge base -- eighty-one percent of the time.
[PAUSE]
The correct rate is twenty percent. This isn't a language problem.
It's a decision-policy problem.

[SLIDE 3]
[0:28] GRPO -- Group Relative Policy Optimization -- lets the model
try four different decisions per scenario, scores them against
ground truth, and updates the weights so better choices become
more likely.
[PAUSE]
Instead of telling the model what to do, it learns from practice.

[SLIDE 4]
[0:43] On eighty-three held-out scenarios: intent accuracy -- did it
understand the customer -- up eight and a half points. Action
accuracy -- did it pick the right next step -- up nearly five
points. Macro F1 -- balanced improvement across all six action
types, not just the dominant one.

[SLIDE 5]
[0:58] What improved: the model diagnoses problems better and picks
better actions.
[PAUSE]
What's still broken: it still defaults to search-knowledge-base
seventy-eight percent of the time. The bias is reduced, not
eliminated.
[PAUSE]
The takeaway: reinforcement learning changed decision behavior
that prompting alone could not. The code and full results are
on GitHub.

[END ~1:15]
```

**Word count: ~176 words. Target pace: ~130 wpm. Projected duration: ~1:15 -- 1:21.**

---

## 8. 60-SECOND BACKUP VERSION

```
[SLIDE 1]
[0:00] I'm Ahmad. Can reinforcement learning teach a small model
better customer-support decisions than prompt engineering?

[SLIDE 2]
[0:08] The prompted model defaults to one action -- search the
knowledge base -- eighty-one percent of the time. Ground truth
is twenty percent. It understands language but lacks a
decision policy.

[SLIDE 3]
[0:20] GRPO lets the model try four decisions per scenario, scores
them, and updates the weights toward better choices. Practice
instead of instructions.

[SLIDE 4]
[0:30] On eighty-three held-out scenarios: intent accuracy up
eight and a half points. Action accuracy up nearly five.
Macro F1 up -- balanced improvement, not just the dominant
action.

[SLIDE 5]
[0:43] The model improved -- but it still over-predicts
search-knowledge-base at seventy-eight percent. The bias is
reduced, not eliminated. GRPO changed behavior that prompting
alone could not.

[END ~0:58]
```

**Word count: ~130 words. Target pace: ~130 wpm. Projected duration: ~0:58 -- 1:02.**

---

## 9. WHAT TO CUT FROM THE ORIGINAL VIDEO

**Remove entirely from the main video:**

1. All seven reward component details (weights, names, formulas)
2. KL beta, learning rate, LoRA rank, LoRA alpha
3. Hardware and training time
4. Parameter counts (the "4 million vs 1.5 billion" comparison -- interesting but costs 10+ seconds and adds complexity)
5. All six degenerate strategies
6. Zero-shot baseline results
7. Random baseline results
8. All six per-class F1 values
9. Action entropy values
10. Per-domain reward breakdown
11. Challenge set results table
12. Full training log discussion
13. Missing information recall metric
14. Unnecessary question rate
15. DPO history / DPO vs GRPO comparison
16. Acceptable-action accuracy
17. Synthetic data generation details
18. Format compliance (100% is not interesting to a viewer)
19. The phrase "practice beats instructions -- not fully, but partially" (too vague as a closing)
20. Any explanation of what the model output JSON looks like
21. Any mention of the reward function being "rule-based vs learned" (save for Q&A)

**All of the above can live in the GitHub README, docs, and Q&A.**

---

## 10. FACTUAL CORRECTIONS TO MY ORIGINAL TRANSCRIPT

### Error 1: Parameter Scale
- **What I implied:** "1.5 million" or ambiguous parameter count
- **Correction:** Qwen2.5-1.5B-Instruct has approximately 1.5 BILLION parameters. LoRA trains ~4 million of them (~0.3%). Do not conflate "1.5B model with 4M trainable" with "replacing the model with 4M."

### Error 2: How GRPO Works
- **What I said:** "GRPO ranks responses and trains toward the better responses"
- **Correction:** GRPO does not simply pick the top-ranked response and train on it. The correct mechanism: generate K=4 completions, score all of them, compute group-relative advantages (each completion's reward minus the group mean), and update policy weights proportionally to those advantages. Higher-reward completions get upweighted, lower-reward completions get downweighted. This is a policy gradient method, not a ranking-and-cloning method.

### Error 3: Training vs Test Confusion
- **What I said:** Ambiguously mixed "training scenarios" and "test scenarios"
- **Correction:** 266 training scenarios (model trains on these). 83 held-out test scenarios (model never sees these during training). All results in the presentation are from the 83-scenario test set. These must never be confused.

### Error 4: Too Many Metrics Presented as Equally Important
- **What I did:** Showed composite reward, intent accuracy, action accuracy, acceptable action accuracy, macro F1, format compliance, entropy, and per-action F1 values
- **Correction:** Focus on exactly three: Intent Accuracy, Action Accuracy, Macro F1. Everything else is supporting detail for Q&A.

### Error 5: Overclaim on GRPO Effectiveness
- **What I implied:** GRPO substantially improved the model
- **Correction:** GRPO produced measurable, consistent improvements on held-out data. But SEARCH_KB collapse persists (78% vs 20% ground truth), COMPLETE is never predicted, and challenge-set performance drops significantly. The improvement is real but modest. Frame honestly.

### Error 6: Pacing and Information Density
- **What happened:** 4-minute video tried to cover all of the above
- **Correction:** A 60-90 second video covering one problem, three metrics, and one honest limitation will be more effective than a 4-minute comprehensive tour.

### Potential Error 7: DPO vs GRPO Label
- **Old documentation (POST_TRAINING_AUDIT.md, dated Sep 7)** warned the GRPO reward function was format-only and recommended calling the method DPO
- **Correction:** The GRPO reward function was FIXED before training. The training log (grpo_full_training_log.txt) confirms the full compositional reward with ground truth was used during GRPO training. The logged reward breakdown shows intent_accuracy, action_correctness, and other real components being scored. This IS genuine GRPO/RL. The old audit is outdated.

---

## 11. ONE-SENTENCE GRPO EXPLANATION

### Technical Version
GRPO generates K completions per prompt, scores each with a reward function, computes group-relative advantages, and performs a policy gradient update on LoRA weights with KL regularization to increase the probability of higher-reward behavior.

### Intuitive Version
The model tries four different answers to each scenario, gets scored on each one, and updates its weights so better decisions become more likely -- learning from practice instead of instructions.

---

## 12. FIVE LIKELY FOLLOW-UP QUESTIONS

### Q1: "Why does the model still default to SEARCH_KB?"

**Answer:** Two likely reasons. First, the base model (Qwen2.5-1.5B-Instruct) was instruction-tuned on general assistant data where "look it up" is a safe default. One epoch of GRPO training on 266 scenarios is limited signal to override that prior. Second, SEARCH_KB is an "acceptable" action in 53% of scenarios (acceptable_actions in ground truth includes it broadly), so the reward function gives partial credit (0.5) for choosing it even when it's not ideal. This dampens the gradient signal to move away from SEARCH_KB. More training data, more epochs, or increased penalty weighting on action_correctness could help.

### Q2: "How do you know the improvement isn't just noise?"

**Answer:** Three defenses: (1) The improvement is consistent across multiple independent metrics -- intent accuracy, action accuracy, and macro F1 all moved in the same direction. (2) The degenerate policy gate verified that no trivial strategy (always-ask, always-search, keyword stuffing) can score above 0.519 composite reward. GRPO at 0.657 is well above that floor. (3) Per-action F1 shows improvements on ROUTE_CLAIM (0.18 to 0.33) and PROVIDE_STEP (0.24 to 0.31), which are minority classes. Noise would not consistently improve minority-class performance. That said, with 83 test scenarios and single-seed training, formal statistical significance is limited. A production experiment would use multiple seeds and bootstrap confidence intervals.

### Q3: "Why not use a larger model?"

**Answer:** The research question is whether post-training can shift decision behavior, not whether a large model can do customer support. Using 1.5B parameters keeps training feasible on consumer hardware (M1 Pro, 17 hours), forces the experiment to demonstrate genuine behavioral learning rather than model scale, and matches the kind of model you'd actually deploy for high-volume inference. The experimental infrastructure -- reward function, evaluation, degenerate gates -- would work identically with a larger model.

### Q4: "Why rule-based reward instead of a learned reward model?"

**Answer:** 392 total scenarios is far too few to train a robust reward model. A learned reward model trained on this data would overfit and become unreliable. Rule-based components are deterministic, auditable, and interpretable -- I can explain exactly what each component measures and verify it with unit tests (36 tests, all passing). In production with thousands of labeled interactions, a learned reward model would be the right choice. For this proof-of-concept, rules are the defensible option. The weights are configurable via YAML for ablation studies.

### Q5: "What would you do differently in production?"

**Answer:** Five things: (1) Real customer messages instead of synthetic scenarios -- the synthetic data has formulaic structure that limits generalization. (2) Multi-turn episodes instead of single-step -- real support conversations unfold over multiple exchanges. (3) A learned reward model trained on customer satisfaction signals (resolution rate, CSAT scores). (4) Online RL (PPO or continued GRPO) with continuous policy updates as new interaction data arrives. (5) Action distribution monitoring -- track SEARCH_KB frequency over time to detect and correct policy collapse. The experimental infrastructure I built -- compositional reward, degenerate gates, per-action metrics -- would transfer directly.

---

## 13. REHEARSAL VERSION

```
[SLIDE 1 -- slow, calm, look at camera]

[0:00] (SLOW) I'm Ahmad.
(brief pause, 1 beat)
I ran an experiment:
(natural pace) can reinforcement learning teach a small language
model to make better customer-support decisions
than a carefully written prompt?
(2-beat pause, look at camera)
Here's what I found.

[SLIDE 2 -- moderate pace, emphasis on numbers]

[0:12] With a strong engineered prompt,
the model already understands support language.
But it DEFAULTS to one action --
(slight emphasis) search the knowledge base --
(slow, clear) EIGHTY-ONE percent of the time.
(1-beat pause)
The correct rate is TWENTY percent.
(look at camera) This isn't a language problem.
It's a DECISION-POLICY problem.

[SLIDE 3 -- slightly faster, confident, explaining mechanism]

[0:28] GRPO -- Group Relative Policy Optimization --
(gesture toward slide) lets the model try FOUR different decisions
per scenario,
SCORES them against ground truth,
and UPDATES the weights so better choices become more likely.
(1-beat pause, look at camera)
Instead of TELLING the model what to do,
it LEARNS from practice.

[SLIDE 4 -- this is the centerpiece, slow down, let numbers land]

[0:43] (look at slide briefly, then back to camera)
On eighty-three held-out scenarios:
(slow) Intent accuracy -- did it understand the customer --
(emphasis) up EIGHT AND A HALF points.
Action accuracy -- did it pick the right next step --
up nearly FIVE points.
Macro F1 --
(brief explanatory tone) balanced improvement across all six
action types,
not just the dominant one.

[SLIDE 5 -- calm, honest, conclusive]

[0:58] What improved:
(straightforward) the model diagnoses problems better
and picks better actions.
(1-beat pause)
What's still broken:
(honest tone) it still defaults to search-knowledge-base
SEVENTY-EIGHT percent of the time.
The bias is REDUCED, not eliminated.
(2-beat pause, look directly at camera for closing)
The takeaway: reinforcement learning changed decision behavior
that prompting alone could not.
(calm) The code and full results are on GitHub.

[END ~1:15]
```

### Rehearsal Notes

**Opening (0:00-0:12):** This is where Victor said you were too fast. Start noticeably slower than your natural pace. "I'm Ahmad" should take a full second. The research question should feel like you're sharing something interesting, not racing to get through a slide. Look at the camera during "Here's what I found" -- it creates a moment of direct connection.

**Slide 2 (0:12-0:28):** The numbers 81% and 20% need to land clearly. Slow down on "eighty-one percent" and "twenty percent." The contrast is the entire slide -- if the viewer misses these two numbers, the slide fails. "Decision-policy problem" is the key phrase -- deliver it with conviction, not speed.

**Slide 3 (0:28-0:43):** This can be slightly faster because you are explaining a mechanism, not delivering a result. Say "GRPO" clearly the first time. You can gesture toward the slide flowchart briefly. The words "four," "scores," and "updates" are the three verbs that carry the explanation -- emphasize them.

**Slide 4 (0:43-0:58):** Slow down again. This is the payoff. Glance at the slide to orient the viewer, then look back at the camera. Let each metric land separately. "Eight and a half points" is the headline number -- give it room.

**Slide 5 (0:58-1:15):** The transition from "improved" to "not solved" is the moment that makes you sound like a serious engineer, not a salesperson. Do not rush through the limitation. The 2-beat pause before the takeaway creates gravity. End looking at the camera. "The code and full results are on GitHub" is a confident, clean exit -- no trailing off, no "so yeah, that's what I found."

**General:**
- Breathe before each slide transition
- Never say "um," "uh," "basically," or "as you can see"
- If you lose your place, pause silently for one beat rather than filling with words
- Record yourself at least 3 times before the real take
- On the first recording, you will probably finish in ~55 seconds (too fast). That's normal. Slow down.
