# RL ARCHITECTURE

---

## 1. System Overview

This project applies GRPO (Group Relative Policy Optimization) to post-train Qwen2.5-1.5B-Instruct on a customer support triage task. The model receives a customer support scenario and outputs a structured JSON decision (intent, action, missing information, reasoning). LoRA adapters (rank 16) are trained via policy gradient against a 7-component rule-based reward function. The system is structured as a contextual bandit: single-step episodes with no sequential dependency.

---

## 2. RL Formulation

### State Space

Defined in `src/environment/state.py:SupportState`:

| Field | Type | Description |
|-------|------|-------------|
| `customer_message` | str | Raw customer text |
| `device_type` | str | Device category (smartphone, tablet, laptop, etc.) |
| `device_name` | str | Specific device model |
| `conversation_turn` | int | Turn number in conversation |
| `available_kb_topics` | list[str] | Knowledge base topics accessible |
| `customer_sentiment` | str | Detected sentiment (frustrated, neutral, etc.) |

### Action Space

Defined in `src/environment/actions.py:VALID_ACTIONS`:

6 discrete actions: `ASK_CLARIFICATION`, `SEARCH_KB`, `PROVIDE_STEP`, `ROUTE_CLAIM`, `ESCALATE`, `COMPLETE`

The model's full output is a JSON with 5 fields: `reasoning`, `intent_assessment`, `missing_information`, `action`, `action_detail`. The action field is the primary decision variable.

### Reward Function

Defined in `src/rewards/reward.py:RewardCalculator`. 7 weighted components:

| Component | Weight | Source |
|-----------|--------|--------|
| action_correctness | 0.25 | `src/rewards/components.py:ActionCorrectnessComponent` |
| intent_accuracy | 0.20 | `src/rewards/components.py:IntentAccuracyComponent` |
| missing_info_recall | 0.15 | `src/rewards/components.py:MissingInfoRecallComponent` |
| clarification_quality | 0.15 | `src/rewards/components.py:ClarificationQualityComponent` |
| efficiency | 0.10 | `src/rewards/components.py:EfficiencyComponent` |
| should_not_penalty | 0.10 | `src/rewards/components.py:ShouldNotPenaltyComponent` |
| format_compliance | 0.05 | `src/rewards/components.py:FormatComplianceComponent` |

When action is not ASK_CLARIFICATION, clarification_quality weight (0.15) is redistributed proportionally to other components.

### Policy

The policy is the LoRA-adapted Qwen2.5-1.5B model's conditional distribution: `P(JSON_output | state, prompt)`. LoRA config: rank=16, alpha=32, dropout=0.05, target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`.

### Episode Structure

Contextual bandit. One state, one action, one reward. No sequential dependency, no discount factor, no value function needed.

---

## 3. Component Architecture

```
src/
  environment/
    state.py          # SupportState, Scenario, GroundTruth dataclasses
    actions.py         # VALID_ACTIONS, VALID_INTENTS, ModelOutput, parse_model_output()
  rewards/
    components.py      # 7 individual reward components
    reward.py          # RewardCalculator: orchestrates components, handles weight redistribution
  policies/
    prompt_templates.py  # SYSTEM_PROMPT_MINIMAL, SYSTEM_PROMPT_ENGINEERED, build_messages()
    inference.py       # PolicyRunner: model.generate() wrapper, extract_json_from_text()
  evaluation/
    metrics.py         # compute_metrics(), confusion matrix, per-action P/R/F1, bootstrap_ci()

scripts/
    generate_scenarios.py  # Synthetic data generation (4-phase pipeline)
    train.py              # GRPO and DPO training loops
    evaluate.py           # Unified evaluation for all conditions
    run_degenerate_gate.py # Degenerate policy verification

configs/
    reward_weights.yaml           # Default weights
    reward_weights_no_efficiency.yaml  # Ablation: efficiency=0
```

### Data Flow (Training)

```
data/processed/train.json
        |
        v
load_scenarios() -> List[Scenario]
        |
        v
prepare_grpo_dataset(scenarios, tokenizer)
  -> Dataset{"prompt": [...], "scenario_id": [...]}
        |
        v
make_grpo_reward_fn(scenarios, reward_calc)
  -> reward_fn closure (captures scenario_map dict)
        |
        v
GRPOTrainer(model, reward_funcs=reward_fn, train_dataset=dataset, peft_config=lora)
        |
        | For each step:
        |   1. Sample batch of prompts
        |   2. Generate K=4 completions per prompt (temperature=0.7)
        |   3. Call reward_fn(prompts, completions, scenario_id=sids)
        |   4. Compute group-relative advantages
        |   5. Policy gradient + KL penalty (beta=0.1)
        |   6. Update LoRA parameters
        v
models/trained_rl/
  adapter_config.json + adapter_model.safetensors
```

### Data Flow (Evaluation)

```
data/processed/test.json
        |
        v
load_scenarios() -> List[Scenario]
        |
        v
load_model_and_tokenizer(condition, device)
  -> base model + optional LoRA merge_and_unload()
        |
        v
PolicyRunner.predict(state) for each scenario
  -> (parsed_json, raw_text)
        |
        v
compute_metrics(results, scenarios, reward_calc)
  -> dict with composite_reward, confusion_matrix, per_action_metrics, etc.
        |
        v
results/{condition}_{split}_results.json
results/{condition}_{split}_predictions.json
```

---

## 4. GRPO Training Loop

Implemented in `scripts/train.py:train_grpo()`:

1. **Prepare dataset:** `prepare_grpo_dataset()` builds a HuggingFace Dataset with `prompt` (tokenized chat template) and `scenario_id` columns from training scenarios.

2. **Build reward closure:** `make_grpo_reward_fn()` creates a closure that captures `scenario_map: Dict[str, Scenario]` and `RewardCalculator`. The closure signature matches TRL's expected `reward_func(prompts, completions, **reward_kwargs)`.

3. **Initialize GRPOTrainer:** Passes model, reward function, dataset, and LoRA config. TRL handles the training loop internals.

4. **For each training step:**
   - Trainer samples a batch (batch_size=4 prompts, but with num_generations=4, each prompt generates 4 completions = 4 unique prompts per step effectively)
   - Model generates K=4 completions per prompt at temperature=0.7
   - Trainer calls `reward_fn(prompts, completions, scenario_id=sids)` — scenario_id comes from the extra dataset column
   - Reward closure: parse JSON -> extract ModelOutput -> look up Scenario via scenario_map -> RewardCalculator.compute() -> return scalar reward
   - Advantages computed: `A_i = r_i - mean(r_1..r_K)` within each group
   - Policy gradient: `loss = -A_i * log P(completion_i | prompt)` + KL penalty
   - LoRA parameters updated via AdamW (lr=1e-5)

5. **Save:** `trainer.save_model()` writes LoRA adapter weights to `models/trained_rl/`.

---

## 5. Reward Function Integration

### The Closure Pattern

```python
def make_grpo_reward_fn(scenarios, reward_calc):
    scenario_map = {s.scenario_id: s for s in scenarios}

    def reward_fn(prompts, completions, scenario_id, **kwargs):
        rewards = []
        for completion, sid in zip(completions, scenario_id):
            parsed = extract_json_from_text(str(completion))
            output = parse_model_output(parsed)
            scenario = scenario_map[sid]
            total, breakdown = reward_calc.compute(output, scenario.ground_truth)
            rewards.append(float(total))
        return rewards

    return reward_fn
```

### How TRL Passes scenario_id

TRL 0.15.2's `GRPOTrainer` (line ~1660 in `grpo_trainer.py`) passes any extra dataset columns as keyword arguments to the reward function. By including `scenario_id` in the dataset, it automatically appears as `reward_kwargs["scenario_id"]`.

### Weight Redistribution

When the model doesn't choose ASK_CLARIFICATION, `clarification_quality` is not applicable. `RewardCalculator` sets that component to 0 and redistributes its weight (0.15) proportionally across the remaining components. This prevents the reward from being artificially depressed for non-ASK actions.

---

## 6. Evaluation Pipeline

`scripts/evaluate.py` supports 5 conditions:

| Condition | Base Model | Adapter | Prompt |
|-----------|-----------|---------|--------|
| random | None | None | None |
| zero_shot | Qwen2.5-1.5B | None | MINIMAL |
| engineered | Qwen2.5-1.5B | None | ENGINEERED |
| dpo | Qwen2.5-1.5B | models/trained_dpo | ENGINEERED |
| rl | Qwen2.5-1.5B | models/trained_rl | ENGINEERED |

For trained conditions (dpo, rl):
1. Load base model
2. Load LoRA adapter via `PeftModel.from_pretrained()`
3. Merge and unload: `model.merge_and_unload()` — folds adapter into base weights
4. Run inference with `PolicyRunner` at temperature=0.3

All conditions use identical: test set, parser, reward weights, metrics computation, temperature. This ensures fair comparison.

### Metrics Computed

- Composite reward (mean, std)
- Intent accuracy, action accuracy (exact + acceptable)
- Per-action precision/recall/F1, macro F1
- Confusion matrix
- Action entropy, action distribution
- Missing info recall/precision
- Unnecessary question rate
- Per-domain reward breakdown

---

## 7. Key Design Decisions

### GRPO over PPO

PPO requires a critic network to estimate V(s). For a contextual bandit (single-step), the critic just learns the mean reward per prompt — no temporal structure to exploit. GRPO computes advantages directly from K completions, eliminating the critic entirely. Fewer parameters, simpler training, same expressiveness for this task.

### LoRA over Full Fine-Tuning

1.5B parameters in fp16 = ~3GB. Full fine-tuning would need optimizer states (~12GB) plus gradients (~3GB). MPS on M1 with 32GB can handle it, but LoRA (4M trainable params, 0.3% of total) is safer — less risk of catastrophic forgetting, faster training, and the base model's language capability is fully preserved.

### Rule-Based over Learned Reward

392 training scenarios is far too few to train a reliable reward model. A learned reward model would memorize the training set. The rule-based reward is:
- Deterministic: same input always gives same score
- Auditable: each component can be inspected independently
- Configurable: weights are in YAML, enabling ablation studies
- No additional model to train or maintain

### 1.5B over Larger Models

- Feasible on consumer hardware (M1 MPS, ~3GB model)
- Forces the experiment to demonstrate genuine learning (can't rely on scale)
- Fast iteration: inference + training in hours, not days
- If post-training works on 1.5B, it will work on larger models
