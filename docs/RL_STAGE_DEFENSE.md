# RL STAGE DEFENSE

**Purpose:** Answers to every question an interviewer might ask about the RL aspects of this project.

---

## 1. "Is this really RL?"

**Yes.** GRPO (Group Relative Policy Optimization) is a policy gradient method. Here's the RL mapping:

| RL Concept | This Project |
|-----------|-------------|
| Agent | Qwen2.5-1.5B with LoRA adapters |
| Environment | Customer support scenario (state) |
| State | Customer message + device + conversation turn + KB topics |
| Action | Structured JSON: intent + action + missing info + reasoning |
| Policy | The model's conditional distribution P(action|state) |
| Reward | 7-component compositional score against ground truth |
| Episode | Single-step (contextual bandit) |
| On-policy | Yes — model generates its own completions at training time |

This is the same family of methods used in DeepSeek-R1 and Qwen's reasoning models.

## 2. "Why GRPO, not PPO?"

PPO requires a critic network (value function) — a second model that estimates expected future reward. For a contextual bandit (single-step episode), there is no "future" — the value function would just learn to predict the mean reward for each prompt. GRPO sidesteps this by computing advantages directly from the group of K completions: `A_i = r_i - mean(r_1..r_K)`. Simpler, fewer hyperparameters, no critic to train.

## 3. "Why not DPO?"

DPO is offline preference optimization, not RL. It generates preference pairs ahead of time and optimizes the model to prefer higher-scoring outputs. The core difference:

- **GRPO (RL):** Model generates completions → scores them → updates weights → generates new completions (on-policy loop)
- **DPO:** Generate all pairs once → optimize offline (no on-policy loop)

GRPO is strictly more powerful because the training distribution tracks the current policy. DPO trains on the base model's output distribution, which becomes stale as the model changes.

We include DPO as an optional comparison branch to answer: "Does online RL outperform offline preference optimization on this task?"

## 4. "What is the reward function?"

7 deterministic, rule-based components. No LLM judge, no learned model.

| Component | Weight | Computation |
|-----------|--------|------------|
| Action correctness | 0.25 | 1.0 if ideal action, 0.5 if acceptable, 0.0 if wrong |
| Intent accuracy | 0.20 | 1.0 if exact match to ground truth intent |
| Missing info recall | 0.15 | Fraction of required info items identified, minus 0.1 per false positive |
| Clarification quality | 0.15 | Keyword overlap between question and scenario topics (redistributed to 0 if ASK not chosen) |
| Efficiency | 0.10 | 1.0 if action is not ASK, or if ASK is ideal; 0.0 if unnecessary ASK |
| Should-not penalty | 0.10 | 0.0 if model chose a forbidden action, 1.0 otherwise |
| Format compliance | 0.05 | 1.0 if output parses as valid JSON with all required fields |

When the model doesn't choose ASK_CLARIFICATION, the clarification_quality weight (0.15) is redistributed proportionally to the other components.

## 5. "How do you prevent reward hacking?"

Three mechanisms:
1. **Efficiency component** — penalizes unnecessary clarification questions. Without this, "always ask" scores ~0.50.
2. **False-positive penalty in missing info** — listing irrelevant items reduces the score.
3. **Degenerate policy gate** — before training, we verify that all 6 trivial strategies score below 0.55.

Known limitation: the clarification quality component uses keyword matching, which is exploitable. In production, I'd replace this with semantic similarity or an LLM judge.

## 6. "How does GRPO training work mechanically?"

Step-by-step for one training step:

1. Trainer takes a batch of prompts from the dataset (batch_size=4, but each prompt generates K=4 completions, so effectively 1 unique prompt per step)
2. Model generates K=4 completions per prompt at temperature=0.7
3. Each completion is passed to the reward function along with its `scenario_id`
4. The reward closure looks up ground truth via `scenario_map[scenario_id]` and calls `RewardCalculator.compute()` to get a scalar reward
5. Within each group of K completions, advantages are computed: `A_i = r_i - mean(r_1..r_K)`
6. Policy gradient: increase log-probability of completions with positive advantage, decrease for negative
7. KL penalty (beta=0.1) prevents the policy from drifting too far from the initial weights
8. LoRA adapter parameters are updated; base model weights stay frozen

## 7. "What is the KL term doing?"

`loss = -advantage * log_prob + beta * KL(policy || reference)`

The reference model is a frozen copy of the initial policy (before any training). The KL penalty prevents the model from:
- Collapsing to a degenerate policy that always picks one action
- Losing its language modeling capability in pursuit of reward
- Drifting so far that it generates unparseable output

beta=0.1 is moderate — strong enough to prevent collapse, weak enough to allow meaningful policy change.

## 8. "Why LoRA?"

- 1.5B model on a laptop GPU (MPS) — full fine-tuning would exhaust memory
- LoRA rank 16 on q/k/v/o projections = ~4M trainable parameters (0.3% of total)
- Base language model knowledge is preserved; only decision behavior changes
- At evaluation time, adapter weights merge into the base model — zero inference overhead

## 9. "Why is the action distribution balanced?"

The original dataset had 74% ASK_CLARIFICATION as the ideal action. A model could score ~0.50 by always asking. After rebalancing:

| Action | Train % | Test % |
|--------|---------|--------|
| ASK_CLARIFICATION | 27% | 28% |
| SEARCH_KB | 22% | 20% |
| PROVIDE_STEP | 21% | 23% |
| ROUTE_CLAIM | 12.5% | 12% |
| COMPLETE | 9.2% | 7.2% |
| ESCALATE | 8.2% | 8.4% |

This forces the model to learn a nuanced policy, not a single default action.

## 10. "Why a contextual bandit, not a multi-turn MDP?"

Customer support triage is a single decision per customer message. The model sees one state and picks one action. There is no sequential dependency — the reward is fully determined by the current (state, action) pair.

Multi-turn modeling would add complexity (credit assignment over conversation turns, discount factors, value function estimation) without adding value for this task. The contextual bandit framing matches the real production use case: triage is a one-shot decision.

## 11. "What would you change at production scale?"

| This MVP | Production |
|----------|-----------|
| 392 synthetic scenarios | Millions of real interactions |
| Rule-based reward | Learned reward model from human preferences |
| Single-step bandit | Multi-turn MDP with delayed reward |
| LoRA on 1.5B | Full fine-tune or LoRA on 7B+ |
| GRPO with K=4 | PPO with rollout buffer |
| Template-based test | A/B testing with real customers |
| Offline evaluation | Online reward tracking + regression monitoring |

## 12. "What if the RL results are negative?"

If the post-trained model doesn't beat the engineered baseline, that's still an informative result:

- **Possible cause 1:** Reward function isn't discriminative enough — "always SEARCH_KB" scores close to correct answers
- **Possible cause 2:** 266 training scenarios is too few for the policy gradient signal to overcome the base model's prior
- **Possible cause 3:** 1 epoch is insufficient — the model needs more passes to shift behavior
- **Possible cause 4:** KL penalty (beta=0.1) is too strong — prevents the policy from changing

Each of these is diagnosable from the training logs and evaluation metrics. The experimental infrastructure makes negative results interpretable, not just failures.

## Labels

```
SAFE LABEL:
  "GRPO — a policy gradient RL method with a rule-based compositional reward"

ACCURATE TECHNICAL DESCRIPTION:
  "Group Relative Policy Optimization with LoRA adapters,
   K=4 rollouts per prompt, 7-component deterministic reward"

LABELS I SHOULD NOT USE:
  - "RLHF" (no human feedback)
  - "PPO" (different algorithm)
  - "reward model" (it's a rule-based function, not a learned model)
  - "fine-tuned on customer data" (synthetic data only)
```
