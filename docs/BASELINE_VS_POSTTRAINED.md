# BASELINE VS POST-TRAINED COMPARISON

**Date:** 2026-09-07
**Status:** NO EXPERIMENTS RUN — all values are NOT MEASURED

---

## Comparison Table

| Metric | Random | Zero-shot (Minimal) | Engineered Prompt | Post-Trained (DPO) | Δ vs Engineered |
|--------|--------|--------------------|--------------------|--------------------|--------------------|
| Composite reward | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Intent accuracy | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Action accuracy (ideal) | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Action accuracy (acceptable) | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Missing info recall | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Missing info precision | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Unnecessary question rate | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Format compliance | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Avg output length (chars) | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Inference time (sec) | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 95% CI (composite) | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

---

## What This Table Should Show After Experiments

The critical comparison is **Engineered Prompt vs Post-Trained**.

A convincing result would show:
1. Post-trained model beats engineered prompt on **composite reward** by > 0.05
2. Post-trained model has higher **intent accuracy** (not just always guessing the most common)
3. Post-trained model has higher **action accuracy on minority actions** (SEARCH_KB, PROVIDE_STEP, ROUTE_CLAIM)
4. Post-trained model has lower **unnecessary question rate** (doesn't always ask)
5. Improvement is outside the **95% confidence interval** of the engineered prompt baseline

An unconvincing result would be:
- Composite reward increases but only because format compliance improved
- Intent accuracy increases but action distribution collapses to always ASK_CLARIFICATION
- Improvement is within noise (overlapping confidence intervals)
- Post-trained model has WORSE performance on minority actions

---

## How to Fill This Table

### Step 1: Run baselines
```bash
cd /Users/ahmadavar/Desktop/aura-lite-posttraining
PYTHONPATH=. python3.11 scripts/run_baselines.py
```

### Step 2: Run DPO training
```bash
PYTHONPATH=. python3.11 scripts/train.py --method dpo --smoke-test  # verify pipeline
PYTHONPATH=. python3.11 scripts/train.py --method dpo              # full training
```

### Step 3: Evaluate trained model (SCRIPT NEEDS TO BE CREATED)
```bash
PYTHONPATH=. python3.11 scripts/evaluate_trained.py
```

### Step 4: Compare
```bash
PYTHONPATH=. python3.11 scripts/compare_results.py
```

**Note:** Steps 3 and 4 scripts do not exist yet. They need to be implemented.

---

## Per-Action Breakdown (TEMPLATE — fill after experiments)

| Action | Test Count | Engineered Acc | Post-Trained Acc | Δ |
|--------|-----------|----------------|-----------------|---|
| ASK_CLARIFICATION | 61 (80%) | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| SEARCH_KB | 11 (14%) | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| PROVIDE_STEP | 3 (4%) | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| ROUTE_CLAIM | 1 (1%) | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| ESCALATE | 0 (0%) | N/A | N/A | N/A |
| COMPLETE | 0 (0%) | N/A | N/A | N/A |

**WARNING:** With only 1 ROUTE_CLAIM and 3 PROVIDE_STEP in test, per-action metrics for minority classes will be statistically unreliable.

---

## Per-Domain Breakdown (TEMPLATE — fill after experiments)

| Domain | Test Count | Engineered Reward | Post-Trained Reward | Δ |
|--------|-----------|-------------------|---------------------|---|
| black_screen | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| charging | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| battery_drain | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| activation | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| wifi_network | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| data_transfer | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| damaged_screen | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| lost_stolen | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| liquid_damage | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| claim_routing | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
