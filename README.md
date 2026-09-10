# Aura Lite: Post-Training a Customer Support Decision Specialist with GRPO

**[Video Demo](https://www.loom.com/share/64a74b534b19478ab98214b40ce81cb1)**

This project applies Group Relative Policy Optimization (GRPO) to post-train a 1.5B-parameter language model as a customer support triage agent. Given a customer message about a device issue, the model selects the correct next action (ask clarification, search knowledge base, provide a troubleshooting step, route a claim, escalate, or mark complete) and diagnoses the underlying problem. GRPO post-training achieved a composite reward of **0.657** vs **0.616** from the best prompt engineering baseline, with the largest gain in intent accuracy (**+8.5 percentage points**), demonstrating that reinforcement learning extracts behavioral signal that instruction-based prompting cannot.

## Results

Evaluated on an 83-scenario held-out test set. All model conditions use the same parser, metrics, and reward weights.

| Metric | Random | Zero-shot | Engineered | **GRPO (RL)** | Delta vs Eng. |
|--------|--------|-----------|------------|---------------|---------------|
| Composite Reward | 0.441 | 0.429 | 0.616 | **0.657** | **+0.041** |
| Intent Accuracy | 12.0% | 0.0% | 57.8% | **66.3%** | **+8.5pp** |
| Action Accuracy (exact) | 21.7% | 25.3% | 30.1% | **34.9%** | **+4.8pp** |
| Action Accuracy (acceptable) | 31.3% | 48.2% | 53.0% | **53.0%** | +0.0pp |
| Macro Action F1 | 0.212 | 0.150 | 0.258 | **0.303** | **+0.045** |
| Format Compliance | 100% | 96.4% | 98.8% | **100%** | +1.2pp |

A degenerate policy gate confirmed no trivial strategy (always-ask, always-search, keyword stuffing) scores above 0.519, establishing a meaningful floor well below the GRPO result.

## How It Works

**Base model:** Qwen2.5-1.5B-Instruct with LoRA adapters (rank 16, alpha 32) on attention projections. Base weights are frozen.

**Training method:** GRPO generates K=4 completions per training prompt, scores each with a compositional reward function, computes group-relative advantages, and updates LoRA weights via policy gradient. KL regularization (beta=0.1) prevents policy collapse.

**Reward function:** 7 deterministic, rule-based components -- action correctness (0.25), intent accuracy (0.20), missing info recall (0.15), clarification quality (0.15), efficiency (0.10), should-not penalty (0.10), format compliance (0.05). No learned reward model -- 392 scenarios is too few for that. Rules are auditable and interpretable.

**Data:** 392 synthetic scenarios (266 train / 43 val / 83 test) across 6 device domains (charging, battery drain, black screen, data transfer, liquid damage, lost/stolen). Zero template leakage between splits.

## Project Structure

```
aura-lite-posttraining/
├── configs/                    # Model, training, and reward weight configs
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── reward_weights.yaml
├── data/
│   ├── raw/scenarios.json      # Full scenario dataset
│   ├── processed/              # Train/val/test splits
│   └── challenge/              # 26-scenario novel-phrasing challenge set
├── docs/                       # Technical write-ups and experiment analysis
│   ├── FINAL_EXPERIMENT_RESULTS.md
│   ├── RL_TECHNICAL_STATEMENT_ONE_PAGE.md
│   ├── RL_ARCHITECTURE.md
│   └── ...
├── models/
│   ├── qwen2.5-1.5b-instruct/  # Base model weights
│   └── trained_rl/              # LoRA adapters (checkpoint-266 = final)
├── results/                    # Evaluation outputs for all conditions
│   ├── rl_test_results.json
│   ├── engineered_test_results.json
│   ├── zero_shot_test_results.json
│   ├── random_test_results.json
│   └── degenerate_gate.json
├── scripts/
│   ├── train.py                # GRPO training loop
│   ├── evaluate.py             # Unified eval for any condition/split
│   ├── run_baselines.py        # Run all baseline conditions
│   ├── run_degenerate_gate.py  # Verify reward function isn't exploitable
│   └── generate_scenarios.py   # Synthetic data generation
├── src/
│   ├── environment/            # Scenario state, actions, episode logic
│   ├── evaluation/             # Metrics computation
│   ├── policies/               # Inference and prompt templates
│   ├── rewards/                # Compositional reward function (7 components)
│   └── training/               # Training utilities
└── tests/                      # Unit tests for reward, environment, degenerate policies
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run evaluation for each condition:

```bash
# Random baseline
PYTHONPATH=. python3.11 scripts/evaluate.py --condition random --split test

# Zero-shot (base model, minimal prompt)
PYTHONPATH=. python3.11 scripts/evaluate.py --condition zero_shot --split test

# Engineered (base model, full engineered prompt)
PYTHONPATH=. python3.11 scripts/evaluate.py --condition engineered --split test

# GRPO post-trained model
PYTHONPATH=. python3.11 scripts/evaluate.py --condition rl --split test

# Challenge set (novel phrasings)
PYTHONPATH=. python3.11 scripts/evaluate.py --condition rl --split challenge
```

Results are written to `results/`.

## Training

Run GRPO post-training:

```bash
PYTHONPATH=. python3.11 scripts/train.py
```

Training configuration is in `configs/training_config.yaml`. Key parameters: 1 epoch, 266 steps, K=4 rollouts per prompt, learning rate 1e-5, KL beta 0.1. The trained LoRA adapters are saved to `models/trained_rl/`.

## Key Files

| File | Description |
|------|-------------|
| `scripts/train.py` | GRPO training loop with on-policy rollouts and KL regularization |
| `scripts/evaluate.py` | Unified evaluation across all conditions and splits |
| `src/rewards/reward.py` | Compositional reward function (7 weighted components) |
| `src/rewards/components.py` | Individual reward component implementations |
| `src/environment/actions.py` | Action space definition and model output parser |
| `src/policies/prompt_templates.py` | System prompts for zero-shot and engineered conditions |
| `src/evaluation/metrics.py` | Per-action F1, confusion matrix, domain-level metrics |
| `configs/reward_weights.yaml` | Reward component weights (ablation-ready) |
| `docs/FINAL_EXPERIMENT_RESULTS.md` | Full results with per-action and per-domain breakdowns |
| `docs/RL_TECHNICAL_STATEMENT_ONE_PAGE.md` | One-page technical summary of the approach |

## Hardware

- **Device:** Apple M1 Pro (MPS backend)
- **Training time:** ~17 hours (1033.6 minutes)
- **Training:** 266 steps, 4 rollouts per step, 1 epoch over 266 scenarios
- **Inference:** ~13.5 seconds per scenario (83 test scenarios in 1119.7 seconds)

## Author

Ahmad Naggayev -- [ahmadavar956@gmail.com](mailto:ahmadavar956@gmail.com)
