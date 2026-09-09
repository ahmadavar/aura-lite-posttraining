# RL METHOD DECISION

**Date:** 2026-09-08

---

## Chosen Method: GRPO (Group Relative Policy Optimization)

### Why GRPO fits this task

1. **On-policy rollouts.** For each prompt, GRPO generates K completions from the current policy, scores them with the reward function, and computes advantages relative to the group mean. This is genuine reinforcement learning — the model's own behavior generates the training signal.

2. **No separate reward model needed.** GRPO uses a rule-based reward function directly. No need to train a reward model first (unlike RLHF with PPO).

3. **KL regularization built in.** GRPOConfig has a `beta` parameter for KL penalty against the reference model, preventing the policy from drifting too far from the pretrained weights.

4. **Library support.** TRL's GRPOTrainer handles rollout generation, reward computation, advantage normalization, and policy gradient updates. We supply the model, reward function, and dataset.

5. **Matches the task structure.** Customer support triage is a contextual bandit — one state, one action, one reward. GRPO's "generate K completions, score, update" loop maps directly to this: K rollouts per scenario, scored by the compositional reward.

### What gets updated

- **LoRA adapters** (rank 16, alpha 32) on `q_proj`, `v_proj`, `k_proj`, `o_proj`
- Base Qwen2.5-1.5B weights are **frozen**
- Reference model is a frozen copy of the initial policy (for KL computation)

### How rollouts are generated

1. GRPOTrainer takes a prompt from the dataset
2. Generates K=4 completions using the current policy (temperature > 0 for diversity)
3. Each completion is a structured JSON decision
4. The reward function parses each completion and scores it against the scenario's ground truth
5. Advantages are computed: `A_i = r_i - mean(r_1..r_K)` (group-relative)
6. Policy gradient update: upweight completions with positive advantage, downweight negative

### How reward is applied

The reward function receives `(prompts, completions, scenario_id, **kwargs)` from GRPOTrainer:

1. For each (prompt, completion, scenario_id) triple:
   - Extract JSON from completion text
   - Parse into ModelOutput
   - Look up scenario ground truth via scenario_id
   - Compute composite reward using RewardCalculator (7 components)
   - Return scalar reward

2. GRPOTrainer normalizes rewards within each group of K completions
3. Policy gradient is computed and applied to LoRA parameters

### Whether a reference model / KL term is used

**Yes.** GRPOConfig `beta=0.1` adds a KL penalty term: `loss = -advantage * log_prob + beta * KL(policy || reference)`. This prevents the model from collapsing to a degenerate policy that maximizes reward at the cost of coherent language.

### Why this is genuinely RL

| RL Criterion | GRPO | Status |
|-------------|------|--------|
| Agent interacts with environment | Model generates completions for scenarios | YES |
| Actions are sampled from policy | Completions are sampled via `model.generate()` with temperature > 0 | YES |
| Reward depends on action quality | RewardCalculator scores correctness vs ground truth | YES |
| Policy parameters are updated from reward signal | LoRA weights updated via policy gradient | YES |
| No expert demonstrations required | Reward function scores any output — no labeled pairs needed | YES |
| Exploration occurs | K=4 diverse completions per prompt | YES |

**GRPO is RL.** It is a policy gradient method with on-policy rollouts and a rule-based reward function. This is the same family of methods used in DeepSeek-R1 and Qwen's reasoning models.

### DPO comparison branch

DPO remains as an optional comparison:
- DPO is **offline preference optimization**, not RL
- It uses the same reward function to generate preference pairs offline
- Useful to compare: "Does online RL (GRPO) outperform offline preference optimization (DPO) on this task?"
- Label honestly: "preference optimization" not "RL"

### Labels

```
SAFE INTERVIEW LABEL:
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
