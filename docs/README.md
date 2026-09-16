# Documentation Index

This project keeps **two generations of documents and does not rewrite the first.**

Documents written *before* the Sep 8–9 training run — the audits, the failure taxonomy, the
empty comparison table — are preserved exactly as written. Where later evidence overtook their
conclusions, a dated `> SUPERSEDED` note was added at the top pointing to the result. Nothing
below those notes was edited.

The reason: pre-run documents are the record of what was predicted before any data existed. An
audit that catches a bug only means something if you can see it was written *before* the fix,
and a failure taxonomy is only a prediction if it pre-dates the failure. Rewriting them after
the fact would destroy the only thing that makes them evidence.

## Written before the run — predictions, audits, plans

| Doc | What it is |
|---|---|
| `BASELINE_VS_POSTTRAINED.md` | Pre-registered comparison plan — success *and* failure criteria, written before any result |
| `FAILURE_ANALYSIS.md` | Pre-registered failure taxonomy — 7 predicted failure modes |
| `REWARD_AUDIT.md` | Reward-function audit; **caught the GRPO reward bug that was then fixed** |
| `POST_TRAINING_AUDIT.md` | Full critical review of the repo before experiments |
| `EXPERIMENT_IMPROVEMENT_PLAN.md` | The P0 fix list produced by those audits |
| `RL_REMEDIATION_MATRIX.md` | Finding-by-finding remediation plan; all CRITICAL items implemented |
| `INTERVIEW_PITCH.md` | Early pitch draft, from when the method was still DPO |
| `PROJECT_STORY.md` | Plain-language narrative, written mid-training |
| `RL_ONE_SLIDE_STORY.md` | Slide layout drafted while training ran |

## Written after the run — measured results

| Doc | What it is |
|---|---|
| `FINAL_EXPERIMENT_RESULTS.md` | **Source of truth.** Full results, per-action and per-domain |
| `RL_TECHNICAL_STATEMENT_ONE_PAGE.md` | One-page technical summary |
| `SLIDE_STORYBOARD.md` | Presentation storyboard with final numbers |
| `SHORT_DEMO_PACKAGE.md` | 75-second video script with final numbers |

## Method and specification — not time-bound

`RL_ARCHITECTURE.md` · `RL_METHOD_DECISION.md` · `RL_STAGE_DEFENSE.md` ·
`SYNTHETIC_DATA_SPEC.md` · `DESIGN_SYSTEM.md`

## Start here

1. `FINAL_EXPERIMENT_RESULTS.md` — what happened
2. `RL_TECHNICAL_STATEMENT_ONE_PAGE.md` — the one-pager
3. `RL_STAGE_DEFENSE.md` — RL design questions answered
