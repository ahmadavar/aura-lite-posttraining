# SLIDE STORYBOARD — Aura Lite Presentation Redesign

---

## SOURCE FACT INVENTORY

### Model & Architecture
- Base model: Qwen2.5-1.5B-Instruct
- Adapter: LoRA rank 16, alpha 32
- Trainable params: 4M (0.3% of total)
- Base weights: frozen

### Training Configuration
- Method: GRPO (Group Relative Policy Optimization)
- Training scenarios: 266
- Epochs: 1
- Rollouts per prompt (K): 4
- Learning rate: 1e-5
- KL beta: 0.1
- Final training reward: 0.787
- Final KL divergence: 0.001
- Training time: 17.2 hours
- Hardware: Apple M1 Pro (MPS backend)

### Dataset
- Total scenarios: 392 (synthetic)
- Train: 266 | Val: 43 | Test: 83
- Challenge set: 26 (novel phrasings)
- Domains: 6 (charging, battery_drain, black_screen, data_transfer, liquid_damage, lost_stolen)
- Challenge set has additional domains: damaged_screen, wifi_network, claim_routing

### Action Space (6 actions)
- ASK_CLARIFICATION — "Can you tell me more?" (need more info)
- SEARCH_KB — "Let me look that up" (search knowledge base)
- PROVIDE_STEP — "Try this fix" (give troubleshooting step)
- ROUTE_CLAIM — "Transferring to claims" (insurance/warranty)
- ESCALATE — "Getting a supervisor" (beyond normal support)
- COMPLETE — "Glad I could help" (problem resolved)

### Reward Function (7 components)
- Action correctness: 25%
- Intent accuracy: 20%
- Missing info recall: 15%
- Clarification quality: 15%
- Efficiency: 10%
- Should-not penalty: 10%
- Format compliance: 5%
- Note: Rule-based (not learned) because 392 scenarios too few for robust reward model

### Experimental Conditions
- Random: No model, no training, no prompt
- Zero-shot: Qwen2.5-1.5B, no training, minimal prompt
- Engineered: Qwen2.5-1.5B, no training, full engineered prompt
- GRPO (RL): Qwen2.5-1.5B + LoRA, GRPO 1 epoch, full engineered prompt
- All evaluated: same parser, same metrics, same reward weights, temperature=0.3

### Degenerate Policy Gate
- ALWAYS_ASK: 0.500
- ALWAYS_SEARCH_KB: 0.491
- ALWAYS_PROVIDE_STEP: 0.455
- ALWAYS_ESCALATE: 0.435
- ALWAYS_COMPLETE: 0.365
- KEYWORD_STUFFER: 0.519
- All below 0.55; engineered baseline = 0.616

### Main Results (83-scenario test set)
| Metric | Random | Zero-shot | Engineered | GRPO | Delta vs Eng. |
|--------|--------|-----------|------------|------|---------------|
| Composite Reward | 0.441 | 0.429 | 0.616 | 0.657 | +0.041 |
| Intent Accuracy | 12.0% | 0.0% | 57.8% | 66.3% | +8.5pp |
| Action Accuracy | 21.7% | 25.3% | 30.1% | 34.9% | +4.8pp |
| Macro F1 | 0.212 | 0.150 | 0.258 | 0.303 | +0.045 |
| Format Compliance | 100% | 96.4% | 98.8% | 100% | +1.2pp |
| Action Entropy | - | 1.254 | 1.012 | 1.162 | +0.150 |

### Per-Action Performance (GRPO)
| Action | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| ESCALATE | 0.80 | 0.50 | 0.62 | 8 |
| SEARCH_KB | 0.26 | 1.00 | 0.41 | 17 |
| ROUTE_CLAIM | 1.00 | 0.20 | 0.33 | 10 |
| PROVIDE_STEP | 0.57 | 0.21 | 0.31 | 19 |
| ASK_CLARIFICATION | 0.50 | 0.09 | 0.15 | 23 |
| COMPLETE | 0.00 | 0.00 | 0.00 | 6 |

### Per-Action F1 Comparison (Engineered vs GRPO)
| Action | Eng. F1 | GRPO F1 | Change |
|--------|---------|---------|--------|
| ROUTE_CLAIM | 0.18 | 0.33 | Recall doubled (0.10 -> 0.20) |
| PROVIDE_STEP | 0.24 | 0.31 | Improved across all metrics |
| ASK_CLARIFICATION | 0.08 | 0.15 | Precision improved (0.33 -> 0.50) |
| SEARCH_KB | 0.38 | 0.41 | Recall reached 100% |
| ESCALATE | 0.67 | 0.62 | Slight decrease (-0.05) |
| COMPLETE | 0.00 | 0.00 | No change (both fail) |

### Per-Domain Reward
| Domain | Engineered | GRPO | Delta |
|--------|-----------|------|-------|
| black_screen | 0.818 | 0.832 | +0.015 |
| liquid_damage | 0.667 | 0.789 | +0.123 |
| lost_stolen | 0.735 | 0.713 | -0.022 |
| data_transfer | 0.728 | 0.706 | -0.022 |
| charging | 0.594 | 0.659 | +0.066 |
| battery_drain | 0.551 | 0.540 | -0.011 |

### Challenge Set (26 scenarios)
| Metric | Test Set (83) | Challenge Set (26) |
|--------|---------------|-------------------|
| Composite Reward | 0.657 | 0.576 |
| Intent Accuracy | 66.3% | 46.2% |
| Action Accuracy | 34.9% | 19.2% |
| Macro F1 | 0.303 | 0.107 |
| Format Compliance | 100% | 100% |

### Key Failures (to preserve honestly)
- SEARCH_KB predicted 78% of the time (ground truth: 20%)
- COMPLETE never predicted (0/6 scenarios)
- Missing info recall: 0%
- Challenge set macro F1 drops from 0.303 to 0.107

---

## CONFLICT CHECK

All numbers internally consistent across slides. No conflicts found.

One note: Slide 5 (Experimental Design) uses "Qwen2.5-1.5B" shorthand; Slide 10 (Training Details) uses "Qwen2.5-1.5B-Instruct" — the full name is correct. Will use full name consistently.

Action accuracy labeled as "Action Accuracy" in main table; GRPO per-action table uses different heading "Per-Action Performance." Both correct but different granularity. No value conflicts.

---

## CURRENT SLIDES -> PROPOSED SLIDES MAPPING

| Current Slide | Content | Redesign |
|---|---|---|
| 1. Title | Title + subtitle + tags | -> Slide 1 (streamlined with pipeline visual) |
| 2. The Problem | 6-action table + challenge statement | -> Slide 2 (transcript card + action blocks) + Slide 3 (SEARCH_KB collapse visual) |
| 3. The Method: GRPO | 4 steps + key components | -> Slide 4 (architecture pipeline) + Slide 5 (GRPO learning visual) |
| 4. The Reward Function | 7-component table | -> Slide 6 (stacked bar / dashboard) |
| 5. Experimental Design | 4-condition table + degenerate gate table | -> Slide 7 (degenerate gate bar chart) + Slide 8 (4 condition cards) |
| 6. Results: GRPO Wins | Full 6-metric table + bullet definitions | -> Slide 9 (4 KPI dashboard) |
| 7. Per-Action Performance | GRPO precision/recall/F1 table | -> Slide 10 (slope chart: Eng vs GRPO F1) |
| 8. What GRPO Improved | F1 comparison table + domain table | -> Slide 11 (failure dashboard) + Slide 12 (domain diverging bars) |
| 9. Generalization: Challenge Set | Test vs challenge table | -> Slide 13 (KPI pairs) |
| 10. Training Details | Parameter table + why-choices bullets | -> Slide 14 (compact tech cards) |
| 11. Final Verdict | What improved / what didn't / tagline | -> Slide 15 (3-block verdict) |

---

## TABLES TO CONVERT TO CHARTS

| Current Table | Proposed Visual |
|---|---|
| 6-action meaning table (Slide 2) | 6 icon/label blocks arranged in a grid |
| Reward components table (Slide 4) | Horizontal stacked 100% bar |
| Degenerate gate table (Slide 5) | Horizontal bar chart with reference line at 0.616 |
| Main results table (Slide 6) | 4 KPI delta cards (highlight only Eng vs GRPO) |
| Per-action P/R/F1 table (Slide 7) | Slope chart (Eng F1 -> GRPO F1 per action) |
| Domain reward table (Slide 8) | Diverging horizontal bar chart around zero |
| Challenge set table (Slide 9) | 4 KPI pairs (test vs challenge) |
| Training config table (Slide 10) | Compact card grid |
| Experimental conditions table (Slide 5) | 4 vertical cards |

---

## PROPOSED SLIDE STORYBOARD (15 slides)

### Slide 1 — TITLE
- **Title:** Aura Lite
- **Subtitle:** Post-Training a Customer Support Decision Specialist with GRPO
- **Visual:** Simple pipeline arrow: Customer Transcript -> Qwen Policy -> Support Action
- **Footer tags:** Qwen2.5-1.5B-Instruct | GRPO | LoRA | Synthetic Environment
- **Key number:** None (title only)
- **Speaker takeaway:** "This project teaches a small language model to make better customer support triage decisions using reinforcement learning."

### Slide 2 — THE DECISION PROBLEM
- **Title:** What is the model deciding?
- **Visual:** Customer transcript card ("My phone screen is black, but it still vibrates.") + 6 action blocks in 2x3 grid
- **Key number:** 6 actions
- **Speaker takeaway:** "This is not open-ended text generation. The model picks one of six specific next actions for every customer message."

### Slide 3 — WHY THE BASE MODEL FAILS
- **Title:** The model understood the language -- not the policy.
- **Visual:** Single dominant bar showing SEARCH_KB at ~80% of predictions. Remaining 5 actions barely visible.
- **Key number:** ~80% SEARCH_KB
- **Speaker takeaway:** "Even with an engineered prompt, the base model defaults to SEARCH_KB for almost everything. It lacks a decision policy."
- **Research question callout:** Can RL post-training teach a more consistent policy than prompting alone?

### Slide 4 — EXPERIMENT ARCHITECTURE
- **Title:** How the system works
- **Visual:** Full vertical pipeline: Synthetic Scenario -> Observable State -> Qwen2.5-1.5B -> 4 Candidate Rollouts -> Ground-Truth Reward -> GRPO -> LoRA Policy Update. Side channel: Hidden ground truth feeds reward only.
- **Key number:** None (architecture)
- **Speaker takeaway:** "The model sees the customer state. The reward function sees the ground truth. GRPO connects them through trial-and-error."

### Slide 5 — HOW GRPO LEARNS
- **Title:** Generate. Score. Compare. Update.
- **Visual:** One scenario -> 4 candidate responses with reward bars (e.g., 0.31, 0.86, 0.42, 0.15). Arrow showing GRPO increases probability of top scorer.
- **Key number:** K=4 rollouts
- **Speaker takeaway:** "For each training prompt, the model tries 4 different answers. It learns from the spread — upweighting good decisions, downweighting bad ones."

### Slide 6 — THE REWARD FUNCTION
- **Title:** The reward is the teacher.
- **Visual:** Horizontal stacked 100% bar showing 7 components with weights. Below: one-line per component explaining what it grades.
- **Key number:** 7 components
- **Speaker takeaway:** "A composite reward grades every response on 7 dimensions. Rule-based because 392 scenarios is too few for a learned reward model."

### Slide 7 — DEGENERATE POLICY GATE
- **Title:** Can the reward be gamed?
- **Visual:** Horizontal bar chart, 6 degenerate policies + vertical reference line at 0.616 (engineered baseline). All bars fall short.
- **Key number:** Best degenerate = 0.519; Engineered baseline = 0.616
- **Speaker takeaway:** "Before training, I verified no trivial strategy scores well. The best cheating strategy gets 0.519. Anything above 0.616 is genuine learning."

### Slide 8 — CONTROLLED EXPERIMENT
- **Title:** Four conditions. One evaluation.
- **Visual:** 4 vertical cards: Random / Zero-shot / Engineered / GRPO. Each shows Model, Training, Prompt. Bottom strip: "Same parser, metrics, reward, temperature, test set." GRPO card highlighted in teal.
- **Key number:** 4 conditions, 83 test scenarios
- **Speaker takeaway:** "We tested four conditions under identical evaluation. The only variable is the training method."

### Slide 9 — HEADLINE RESULT
- **Title:** GRPO improved every primary metric.
- **Visual:** 4 large KPI cards: Composite Reward (0.616 -> 0.657, +0.041), Intent Accuracy (57.8% -> 66.3%, +8.5pp), Action Accuracy (30.1% -> 34.9%, +4.8pp), Macro F1 (0.258 -> 0.303, +0.045). Labeled "Engineered vs GRPO".
- **Key number:** +8.5pp intent accuracy (largest gain)
- **Speaker takeaway:** "The RL-trained model beats the best prompt on every aggregate metric. The biggest win is intent accuracy — the model got 8.5 percentage points better at diagnosing problems."

### Slide 10 — WHERE DID THE POLICY CHANGE?
- **Title:** Per-action F1: Engineered vs GRPO
- **Visual:** Slope chart or paired horizontal bars. 6 actions, left column = Eng F1, right column = GRPO F1. Lines connecting them. Green for improvements, orange for regressions.
- **Key numbers:** ROUTE_CLAIM 0.18->0.33, PROVIDE_STEP 0.24->0.31, ESCALATE 0.67->0.62, COMPLETE 0.00->0.00
- **Speaker takeaway:** "GRPO improved the rare actions — ROUTE_CLAIM F1 nearly doubled. ESCALATE slightly regressed. COMPLETE remains unsolved."

### Slide 11 — FAILURE DASHBOARD
- **Title:** What GRPO did not solve.
- **Visual:** 3 warning cards: (1) SEARCH_KB still 78% of predictions (ground truth: 20%), (2) COMPLETE: 0/6 detected, (3) Missing info recall: 0%. Orange accent.
- **Key number:** 78% SEARCH_KB (vs 20% ground truth)
- **Speaker takeaway:** "The model still over-predicts SEARCH_KB. It never predicts COMPLETE. It never detects missing information. These are honest limitations."

### Slide 12 — DOMAIN-LEVEL IMPACT
- **Title:** Where did GRPO help most?
- **Visual:** Diverging horizontal bar chart around zero. Positive (teal): liquid_damage +0.123, charging +0.066, black_screen +0.015. Negative (orange): lost_stolen -0.022, data_transfer -0.022, battery_drain -0.011.
- **Key number:** liquid_damage +0.123 (largest domain gain)
- **Speaker takeaway:** "Biggest improvement on liquid damage and charging. Small regressions on lost/stolen and data transfer."

### Slide 13 — GENERALIZATION TEST
- **Title:** Novel phrasings remain difficult.
- **Visual:** 4 KPI pairs: Test Set vs Challenge Set for Reward (0.657/0.576), Intent (66.3%/46.2%), Action Accuracy (34.9%/19.2%), Macro F1 (0.303/0.107). Format compliance: 100% on both.
- **Key number:** Challenge reward 0.576 > random 0.441 (still above floor)
- **Speaker takeaway:** "Performance drops on novel phrasings, as expected. But the model still beats random, and format compliance stays perfect. Generalization needs more diverse training data."

### Slide 14 — TRAINING CONFIGURATION
- **Title:** Built on consumer hardware.
- **Visual:** Compact card grid: Model (Qwen2.5-1.5B-Instruct), Method (GRPO), Adapter (LoRA, 4M params, 0.3%), Data (266 scenarios), Rollouts (K=4), Epochs (1), LR (1e-5), KL Beta (0.1), Time (17.2h), Hardware (M1 Pro MPS).
- **Key number:** 4M trainable parameters (0.3% of 1.5B)
- **Speaker takeaway:** "Trained on a MacBook. LoRA adapters — only 0.3% of the model was actually updated. The rest stays frozen."

### Slide 15 — FINAL VERDICT
- **Title:** Practice beat instructions -- but only partially.
- **Visual:** 3 blocks: WHAT IMPROVED (teal) / WHAT DID NOT (orange) / NEXT STEPS (navy). Bullet points in each. Bottom: "GRPO post-training extracted behavioral signal that prompting could not."
- **Key numbers:** +8.5pp intent, +0.045 macro F1
- **Speaker takeaway:** "RL improved the policy. It didn't solve everything. Here's what I'd do next: more data, rebalance the reward, multi-turn episodes."

---

## ITEMS TO REMOVE FROM CURRENT DECK

- Slide 5 current: experimental conditions table + degenerate gate table on same slide (too dense). Split into 2 slides.
- Slide 6 current: bullet list explaining what each metric means (move to speaker notes, not slide text).
- Slide 8 current: two tables on one slide (F1 comparison + domain reward). Split into 2 slides.
- All "Why rule-based?" paragraphs: condense to one sentence on reward slide.
- Verbose bullet text throughout: replace with short headlines + visuals.
