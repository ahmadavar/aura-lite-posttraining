# INTERVIEW PITCH & DEFENSE

**Date:** 2026-09-07
**Status:** Framework ready — fill in actual numbers after experiments

---

## PHASE 12 — INTERVIEW DEFENSE ANSWERS

### 1. "Why did you choose this task?"

"Customer support triage is a high-frequency, structured decision problem — the agent sees a message and picks from a discrete action set. That structure makes it measurable. Unlike open-ended generation, I can objectively score whether the model picked the right action, identified the right intent, and asked the right questions. That measurability is what makes post-training tractable."

### 2. "Why is this a good post-training problem?"

"The base model already understands customer support language — it can parse complaints, it knows what 'charging issue' means. What it lacks is a consistent task POLICY: when to ask vs. search vs. escalate. Post-training is specifically designed to shape behavior, not inject knowledge. This task isolates that behavioral dimension."

### 3. "Why not use a frontier model?"

"A frontier model (GPT-4, Claude) would likely score higher out of the box. But the research question isn't 'can a big model do customer support?' — it's 'can post-training shift a small model's decision behavior?' Using 1.5B parameters makes the training feasible on a laptop GPU and forces the experiment to demonstrate genuine learning, not just scale."

### 4. "Why not just use retrieval (RAG)?"

"RAG solves the knowledge problem — 'what are the troubleshooting steps for a black screen?' But this task is about the DECISION problem — 'given what I know, should I ask more questions, search, or act?' RAG doesn't change the model's decision policy. Post-training does."

### 5. "Why not fine-tune supervised?"

"Supervised fine-tuning requires labeled (input, correct_output) pairs — essentially expert demonstrations. I don't have expert demonstrations. What I have is a reward function that can SCORE any output. DPO lets me use that reward to generate preference pairs from the model's own outputs — no expert labels needed. This is closer to how you'd operate in production, where you have rubrics but not labeled data."

### 6. "Why use DPO / preference optimization?"

"DPO is offline preference optimization — it turns reward scores into pairwise preferences and optimizes the model to prefer higher-scoring outputs. It's more stable than online RL (PPO/GRPO) for small datasets, doesn't require training a separate reward model, and the implementation is well-supported in the TRL library. For a proof-of-concept with 194 training scenarios, DPO is the pragmatic choice."

### 7. "How did you design the reward?"

"The reward is compositional — 7 weighted components: intent accuracy (0.20), action correctness (0.25), missing info recall (0.15), clarification quality (0.15), efficiency penalty (0.10), forbidden action penalty (0.10), and format compliance (0.05). Each component is deterministic and rule-based — no LLM judge, no learned reward model. The weights are configurable via YAML so I can run ablations."

### 8. "How did you prevent reward hacking?"

"Three mechanisms: First, the efficiency component penalizes unnecessary clarification questions — without it, the model could always ask and score well. Second, the false-positive penalty in missing-info recall punishes listing irrelevant items. Third, the should-not component gives zero reward for explicitly forbidden actions. I also plan a reward ablation — training with and without the efficiency penalty — to demonstrate that the reward design actually affects learned behavior."

*Note: Be honest about limitations — keyword-based clarification quality IS exploitable. If asked, say "I acknowledged that the clarification evaluator uses keyword matching, which could be gamed. In production I'd use a semantic similarity measure or an LLM judge. For this proof-of-concept, it's a known limitation I documented."*

### 9. "How did you evaluate improvement?"

"Frozen test set of 76 scenarios, never seen during training. Same parser, same metrics, same temperature for all conditions. I compare against three baselines: random, zero-shot (weak prompt), and engineered prompt (strong prompt). The critical comparison is engineered prompt vs post-trained — that isolates what training adds beyond prompt engineering."

### 10. "How do you know improvement isn't overfitting?"

"Three checks: First, the test set has zero template overlap with training data — I split at the template level, not scenario level. Second, I report bootstrap confidence intervals. Third, I look at per-action performance: if improvement only shows up on the majority class (ASK_CLARIFICATION), that's suspicious. Real improvement should show up in minority action accuracy."

*If results show overfitting, be honest: "The test set performance was [X], which is [lower than/similar to] training performance. Given the limited data size (194 training scenarios), some overfitting is expected. With more data, I'd expect the gap to narrow."*

### 11. "What failed?"

*Fill in after experiments. Template:*

"The initial GRPO implementation used a reward function that only checked format compliance — it couldn't distinguish correct from incorrect decisions. I caught this in an internal audit and switched to DPO with the full compositional reward. [If applicable: The model initially collapsed to always picking ASK_CLARIFICATION, so I adjusted the efficiency weight to penalize unnecessary questions more heavily.]"

### 12. "What would you do with production data?"

"Production data would change three things: First, real customer messages instead of synthetic templates — much more diverse, much harder. Second, human expert labels for ground truth actions — replacing my rule-based reward with actual outcome data. Third, online learning where the model generates actions in real conversations and receives delayed reward from resolution outcomes. The architecture I built — state, action, reward components — would carry over directly."

### 13. "What would change at millions of interactions?"

"At scale, four things change: (1) I'd switch from DPO to online RL (PPO or GRPO) because I'd have enough interaction data for on-policy learning. (2) I'd train a learned reward model instead of rule-based components — human preferences would teach the model what 'good' looks like. (3) I'd need to worry about distribution shift and reward hacking at scale — the model could find exploits in the reward function that wouldn't surface in 314 scenarios. (4) The evaluation would shift from accuracy to business metrics: resolution rate, customer satisfaction, escalation rate."

### 14. "How would you monitor regression in production?"

"Three-layer monitoring: (1) Format compliance rate — if it drops, the model is producing unparseable outputs, immediate rollback. (2) Action distribution monitoring — if one action's frequency shifts more than 2 standard deviations from baseline, flag for review. (3) Reward component tracking — if any individual reward component drops (e.g., intent accuracy falls while action accuracy stays flat), that signals a specific behavioral regression. All of these are computable in real-time without human labels."

---

## PHASE 13 — WHY NARROW SPECIALIST > GENERIC CHATBOT

### Is the narrow specialist approach better?

**Yes, IF the repository supports it. Current evidence:**

| Claim | Supported? | Evidence |
|-------|-----------|----------|
| Narrower behavior is easier to measure | **YES** | 7 discrete, computable reward components |
| Action correctness is objectively evaluable | **YES** | 6 discrete actions with ground truth labels |
| Reward decomposes into atomic criteria | **YES** | Component weights are configurable and ablatable |
| Baseline comparison is cleaner | **YES** | Same model, same prompt, only training differs |
| Failure modes are classifiable | **YES** | Wrong intent, wrong action, unnecessary ask, etc. |
| Small specialist can complement frontier orchestrator | **CONCEPTUAL** | Not demonstrated in code |
| Task resembles repeated production decisions | **YES** | Contextual bandit framing matches real support triage |
| Experiment answers a falsifiable question | **YES** | "Does DPO improve action accuracy over engineered prompt?" is testable |

**The narrow specialist framing is the project's strongest asset.** It turns a vague "trained a chatbot" into a specific "shaped decision behavior on a measurable task."

---

## PHASE 16 — PITCHES

### A. 15-second explanation

"I post-trained a small language model to make better customer support triage decisions. Using a compositional reward function, I showed that preference optimization improves action selection accuracy beyond what prompt engineering alone achieves."

*[Adjust based on actual results. If results are weak: "I built the full experimental pipeline to test whether post-training improves customer support triage decisions, and I'll walk you through what I learned about reward design and evaluation."]*

### B. 30-second explanation

"This is a post-training experiment on a narrow task: customer support triage. A 1.5B parameter model receives a customer message and must choose the right next action — ask a question, search the knowledge base, escalate, or route a claim. I designed a compositional reward function with 7 components — intent accuracy, action correctness, information gap detection, and efficiency penalties. I used DPO to generate preference pairs from the model's own outputs, scored by the reward, and trained LoRA adapters. The key finding is [FILL: what actually happened]."

### C. 60-second technical explanation

"The hypothesis is that post-training a small model on a narrow, repeated decision task produces better behavior than prompting alone. The task is customer support triage — structured as a contextual bandit where the model observes a support state and selects from 6 discrete actions. The reward function has 7 components — the biggest weight goes to action correctness at 0.25, followed by intent accuracy at 0.20. I used DPO because I don't have expert demonstrations, but I do have a reward function. For each training scenario, I generate 4 completions, score them with the reward, and create preference pairs from the best and worst. LoRA rank 16 on attention projections. I compare against three baselines: random, minimal prompt, and the same engineered prompt used during training. The critical comparison is engineered prompt vs post-trained — that's what isolates the effect of training. [FILL: Results. If positive: 'The post-trained model improved action accuracy by X% on the held-out test set, with the biggest gain on minority actions like SEARCH_KB.' If negative: 'I found that the model tended to collapse toward always asking clarification, which taught me about reward hacking and the importance of the efficiency penalty.']"

### D. 2-minute deep-dive explanation

"Let me walk you through the full experiment.

**Problem:** Customer support triage requires making a decision — ask, search, escalate, route, or resolve. This is a repeated, high-frequency decision that a small specialist model could handle.

**Why post-training:** The base model — Qwen 2.5, 1.5 billion parameters — already understands customer support language. The gap isn't knowledge, it's behavioral consistency. Post-training is specifically designed to shape behavior.

**Environment:** I structured this as a contextual bandit — single-step episodes. The model sees the customer message, device type, conversation turn, and available knowledge base topics. Ground truth is hidden. It outputs a JSON with reasoning, intent assessment, missing information list, action, and action detail.

**Reward function:** 7 weighted components. Intent accuracy — did you identify the right problem? Action correctness — did you pick the right next step? Missing info recall — did you identify what's still unknown? Clarification quality — if you asked a question, was it relevant? Efficiency — did you ask when you didn't need to? Forbidden action — did you avoid actions that would harm the customer? Format — is the output parseable? Each component is deterministic, rule-based, and configurable.

**Training:** DPO with LoRA adapters. For each of 194 training scenarios, I generate 4 completions from the base model, score each with the reward function, and create preference pairs from the highest and lowest scoring outputs. The model learns to prefer outputs that the reward function scores higher.

**Evaluation:** 76 frozen test scenarios with zero template overlap from training. Three baselines — random, weak prompt, strong engineered prompt. Same parser, same metrics, same temperature for all conditions.

**[FILL: Results and what you learned. Always end with the honest learning, not just the numbers.]**

**What I'd do differently:** A semantic evaluator instead of keyword matching for clarification quality. More diverse test scenarios to test generalization. And with production data, I'd switch from rule-based rewards to a learned reward model trained on customer satisfaction outcomes."

---

## PHASE 17 — "WHY POST-TRAINING?" ANSWER

### Template (fill in after experiments)

**If results are positive:**

"The base model with our best prompt already achieved [X]% action accuracy. But it consistently made the same mistakes — [SPECIFIC EXAMPLE: e.g., 'it always asked for more information on claim routing scenarios instead of routing directly']. Post-training didn't teach the model new facts about customer support. It shifted the model's decision boundary — after training, it [SPECIFIC BEHAVIORAL CHANGE]. That's what post-training buys you: not knowledge, but behavior."

**If results are negative or mixed:**

"Post-training on this dataset didn't produce the improvement I expected, and that itself is an interesting finding. The model tended to [DESCRIBE FAILURE MODE]. This suggests that [INSIGHT — e.g., 'the reward function's ASK_CLARIFICATION bias made it too easy to score well without actually improving decisions' or 'with only 194 training scenarios, there wasn't enough signal to overcome the base model's prior']. What I learned is that the reward design and data diversity are the hard parts of post-training — not the optimization algorithm."

---

## PHASE 18 — ONE SLIDE RESULT STORY

### Layout

```
╔══════════════════════════════════════════════════════════════╗
║  Post-Training a Customer Support Decision Specialist       ║
╠═══════════════════════════╦══════════════════════════════════╣
║  METRICS                  ║  BEHAVIORAL EXAMPLE              ║
║                           ║                                  ║
║  Condition  | Reward | Acc║  Scenario: "My screen cracked,   ║
║  ──────────────────────── ║   need to get it fixed"          ║
║  Random     | [X]   | [X]║                                  ║
║  Zero-shot  | [X]   | [X]║  BASELINE:                       ║
║  Engineered | [X]   | [X]║  → ASK_CLARIFICATION             ║
║  Post-train | [X]   | [X]║  "Is there visible damage?"      ║
║                           ║                                  ║
║  Δ vs engineered: +[X]    ║  POST-TRAINED:                   ║
║                           ║  → ROUTE_CLAIM                   ║
║                           ║  "Routing to claims for screen   ║
║                           ║   replacement assessment"        ║
║                           ║                                  ║
╠═══════════════════════════╩══════════════════════════════════╣
║  TAKEAWAY: Post-training improved task policy, not general   ║
║  language knowledge. The model learned WHEN to act vs ask.   ║
╚══════════════════════════════════════════════════════════════╝
```

*Replace [X] with actual numbers. Replace example with a real observed behavior change.*

*If results are negative, change takeaway to: "Post-training exposed reward hacking: the model exploited the ASK_CLARIFICATION bias rather than learning diverse actions. This demonstrates why reward design is the hard part of post-training."*

---

## SECURITY / CONFIDENTIALITY CHECK

| Item | Status |
|------|--------|
| Proprietary company info | NONE — all synthetic |
| Internal slides/screenshots | NOT IN REPO |
| Real customer data | NONE — synthetic templates |
| API keys / secrets | NONE |
| Company architecture details | NONE — project is "inspired by" not "recreation of" |
| Names that should be private | NONE in code (Asurion mentioned only in past conversation, not in repo) |

**Safe positioning:** "A synthetic experimental analogue of a customer-support post-training problem."

**DO NOT say:** "I recreated Asurion's AURA system" or "This uses Asurion's data/architecture."
**DO say:** "Inspired by the customer AI post-training work described in a practicum presentation."
