# ONE SLIDE RESULT STORY

**Status:** Baselines complete, RL training in progress

---

## Layout

```
+==============================================================+
|  Post-Training a Customer Support Decision Specialist         |
|  GRPO with Compositional Reward on Qwen2.5-1.5B              |
+==============================================================+
|                           |                                   |
|  RESULTS (83-scenario     |  THE PROBLEM                      |
|  test set)                |                                   |
|                           |  Base model collapses to one      |
|  Condition  | Reward | F1 |  action (SEARCH_KB ~80%) even     |
|  ─────────────────────────|  with a strong prompt.            |
|  Random     | 0.441 | .21 |                                   |
|  Zero-shot  | 0.429 | .15 |  Can RL post-training diversify   |
|  Engineered | 0.616 | .26 |  the policy and improve           |
|  GRPO (RL)  | 0.657 | .30 |  decision accuracy?               |
|                           |                                   |
|  Δ vs engineered: +0.041  |  METHOD                           |
|                           |  GRPO: 4 rollouts/prompt,         |
|  DEGENERATE GATE          |  7-component rule-based reward,    |
|  Best trivial: 0.519      |  LoRA adapters (rank 16),          |
|  (all < 0.55)             |  1 epoch on 266 scenarios          |
|                           |                                   |
+===========================+===================================+
|  TAKEAWAY                                                     |
|  RL post-training improved every metric over the best prompt. |
|  Intent accuracy +8.5pp, macro F1 +0.045, reward +0.041.     |
|  The model learned better problem diagnosis and more balanced  |
|  action selection. Practice beat instructions.                 |
+===============================================================+
```

## Key talking points for each cell

### Results table
- "Random baseline gives you the floor — 0.441 reward, 0.21 macro F1"
- "Zero-shot is worse than random on F1 because it collapses harder to one action"
- "Engineered prompt is the benchmark — 0.616 reward. Anything the RL model does above that is what training bought us"
- "Macro F1 is the honest metric — it punishes action collapse"

### The problem
- "The base model knows customer support language. It lacks a consistent decision POLICY."
- "Even with a detailed prompt, it picks SEARCH_KB 80% of the time"
- "This is the gap post-training is supposed to close"

### Method
- "GRPO is a policy gradient method — same family as DeepSeek-R1"
- "4 rollouts per prompt means the model tries 4 different answers, learns from the spread"
- "Rule-based reward, not a learned reward model — 392 scenarios is too few for a learned model"
- "LoRA keeps the base model frozen — only 4M trainable parameters out of 1.5B"

### Degenerate gate
- "Before training, I verified no trivial strategy can score above 0.55"
- "This means any improvement above 0.616 is genuine, not a shortcut"

### Takeaway
- Frame positively if results are positive
- Frame as a learning about reward design if negative
- Either way, the experimental methodology is sound
