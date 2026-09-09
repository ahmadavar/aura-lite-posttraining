# The Story of This Project

---

## The Problem I Was Trying to Solve

Imagine you're running a customer support team. Every time a customer reaches out — "my phone screen is cracked" or "my battery keeps dying" — an agent has to make a quick decision: Should I ask the customer more questions? Should I look up a troubleshooting guide? Should I route this to the claims department? Should I escalate to a senior agent?

Humans learn this judgment through experience. After handling hundreds of calls, a support agent develops an instinct for when to ask versus when to act. The question I asked was: **can a small AI model learn that same judgment through practice?**

## What I Built

I took a 1.5-billion parameter language model (Qwen2.5) — small enough to train on a laptop — and gave it a practice environment. The model sees a customer support scenario and has to pick from 6 possible actions:

1. Ask a clarifying question
2. Search the knowledge base
3. Provide a troubleshooting step
4. Route to claims
5. Escalate to a senior agent
6. Mark the case as resolved

For each scenario, I know what the "right" answer is. So I built a scoring system — a reward function with 7 different criteria — that tells the model how well it did. Did it pick the right action? Did it correctly identify the customer's problem? Did it ask a useful question, or a pointless one?

## What the Model Does Before Training

Before any training, I tested the model three ways:

**Random guessing** — Pick an action at random. Score: 0.44 out of 1.0. This is the floor.

**Minimal prompt** — Just tell the model "you're a support agent, pick an action." Score: 0.43. Barely better than random. The model collapses — it picks "search the knowledge base" for almost every single scenario, regardless of context.

**Detailed prompt** — Give the model a long, carefully written prompt explaining exactly how to think about each scenario. Score: 0.62. Much better. But the model still picks "search the knowledge base" about 80% of the time. It understands the task better, but it hasn't learned a diverse decision policy.

The detailed prompt gets you from 44% to 62% of the maximum score. That's a meaningful jump — but the model still makes the same mistake over and over. It's like a new support agent who learned the rule "when in doubt, look it up" and applies it to everything.

## How the Training Works

This is where reinforcement learning comes in. Instead of showing the model the right answers (that would be supervised learning), I let it practice.

For each support scenario, the model tries 4 different responses. My scoring system grades each one. The model then learns: "Response #2 scored highest — I should do more of that. Response #4 scored lowest — I should do less of that."

This is the same basic idea behind how ChatGPT and Claude were trained to follow instructions. The technical name is GRPO — Group Relative Policy Optimization. The model generates a group of attempts, compares them, and updates its behavior toward the better ones.

I set up 266 practice scenarios. The model practices on each one, generating 4 attempts per scenario and learning from the spread. Only the "adapter" weights get updated — about 4 million parameters out of 1.5 billion. Think of it like fine-tuning the model's decision-making wiring without touching its language understanding.

## What I'm Looking For

The key comparison is simple: **does the trained model beat the best prompt?**

The best prompt scores 0.62. If the trained model scores higher, that means training taught the model something that a clever prompt couldn't. Specifically, I'm looking for:

- **Higher overall score** — does the model make better decisions?
- **More diverse actions** — does it stop defaulting to "search" and start routing claims, escalating dangerous situations, and completing resolved cases?
- **Better minority-action performance** — the hardest test is whether the model learns to recognize the less common situations (claim routing, escalation) rather than just getting better at the majority case

## What Makes This Legitimate

A few things I built to keep myself honest:

**Degenerate policy gate.** Before training, I tested what happens if the model just picks the same action every time. "Always ask" scores 0.50. "Always search" scores 0.49. No trivial strategy scores above 0.55. So if the trained model scores above 0.62, it's doing something non-trivial.

**Balanced practice scenarios.** My original dataset was 74% "ask a question" scenarios — meaning the model could score well by always asking. I rebalanced to ~27% ask, with meaningful representation of all 6 actions.

**Confusion matrix.** Instead of just reporting one accuracy number, I track what the model predicts for every action type. This shows whether the model actually learned to distinguish between situations, or just found a new default behavior.

**Challenge set.** 26 hand-written scenarios with angry customers, multi-issue complaints, contradictory information, and situations where the customer already tried the obvious steps. These test whether the model generalizes or just memorized patterns.

## The Scoring System (Reward Function)

The scoring system has 7 parts, each measuring something different:

- **Did you pick the right action?** (25% of the score) — The most important thing. Did you route the claim, or did you ask an unnecessary question?
- **Did you identify the right problem?** (20%) — "Charging issue" vs "screen damage" vs "data loss"
- **Did you spot what information is missing?** (15%) — If the customer didn't mention their device model, did you notice?
- **If you asked a question, was it a good one?** (15%) — "What device do you have?" is useful. "Can you tell me more?" is generic.
- **Did you ask when you didn't need to?** (10%) — Penalizes unnecessary questions when you have enough info to act
- **Did you avoid harmful actions?** (10%) — Don't tell someone with a safety issue to reboot their phone
- **Can the system read your output?** (5%) — The model must output structured JSON, not free text

These weights are configurable — I can remove any component and retrain to test whether it matters. That's the ablation study.

## What Happens Next

The model is training right now (~3 hours on my laptop). Once it finishes, I'll evaluate it on the 83-scenario test set (which it never saw during training) and the 26-scenario challenge set. Then I'll fill in the final column of the results table and write up what happened — positive or negative.

If it works, the story is: **"RL post-training shifted a small model's decision policy beyond what prompt engineering could achieve."**

If it doesn't work, the story is still interesting: **"Here's why reward design and data diversity are the hard parts of post-training, not the optimization algorithm."**

Either way, the infrastructure — reward function, evaluation pipeline, degenerate gates, metrics — is what makes this a credible experiment rather than a demo.
