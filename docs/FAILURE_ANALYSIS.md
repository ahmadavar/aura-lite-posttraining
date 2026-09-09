# FAILURE ANALYSIS

**Date:** 2026-09-07
**Status:** Pre-experiment — categories and framework defined, no real examples yet

---

## Predicted Failure Categories

Based on code analysis and reward structure, these are the expected failure modes:

### Category 1: Always ASK_CLARIFICATION (Reward Hacking)

**What happens:** Model collapses to always choosing ASK_CLARIFICATION regardless of scenario context.

**Why it would happen:** ASK_CLARIFICATION is ideal/acceptable in ~85% of scenarios. The reward function gives 0.5 (acceptable) for ASK_CLARIFICATION even when another action is ideal. The "always ask" strategy has a high reward floor (~0.55).

**How to detect:** Check action distribution of post-trained model. If ASK_CLARIFICATION > 90%, this has occurred.

**Expected baseline behavior:** Engineered prompt should already show some diversity (SEARCH_KB, PROVIDE_STEP, ROUTE_CLAIM appear in the action list).

**Expected post-trained behavior:** If reward hacking occurs, action diversity DECREASES after training.

**Whether training improved it:** NOT MEASURED

### Category 2: Wrong Intent, Right Action

**What happens:** Model correctly chooses ASK_CLARIFICATION but predicts wrong intent. Intent accuracy stays low while action accuracy is high.

**Why it would happen:** Intent requires understanding the specific problem domain. Action is dominated by one class (ASK_CLARIFICATION). A model can score well on R_action without understanding intent.

**How to detect:** Compare intent accuracy vs action accuracy. If action accuracy >> intent accuracy, this pattern exists.

**Whether training improved it:** NOT MEASURED

### Category 3: Keyword Stuffing in Clarification

**What happens:** Model outputs a generic clarification question stuffed with topic keywords to maximize R_clarification score.

**Example:** "Can you tell me about any physical damage, recent events, drops, water exposure, battery health, cable condition, and insurance status?"

**Why it would happen:** R_clarification uses keyword matching. Including more topic words = higher score, regardless of whether the question is specific or useful.

**How to detect:** Check if clarification questions become longer and more generic after training. Look for many topic keywords in a single question.

**Whether training improved it:** NOT MEASURED

### Category 4: Missing Information List Exploitation

**What happens:** Model outputs a long list of common missing information items to maximize recall.

**Why it would happen:** R_missing_info rewards recall with -0.1 FP penalty. If the model learns that common items (physical_damage_check, recent_events, battery_health_percent) hit many scenarios, it could always include them.

**How to detect:** Check average length of missing_information lists before/after training. If lists grow significantly, this is occurring.

**Whether training improved it:** NOT MEASURED

### Category 5: Template Memorization

**What happens:** Model memorizes (message pattern → correct output) from training data, but fails on novel phrasings at test time.

**Why it would happen:** 126 unique training messages with LoRA rank 16 — enough capacity to memorize patterns. Template-based generation means test messages are stylistically similar.

**How to detect:** Compare performance on "clear" vs "ambiguous" vs "misleading" scenarios. If clear scenarios (most similar to templates) have much higher scores, memorization is likely.

**How to test:** Create 10-20 novel customer messages not following template patterns. Evaluate separately.

**Whether training improved it:** NOT MEASURED

### Category 6: Parser Failure on Post-Trained Output

**What happens:** Post-trained model generates text that the JSON parser cannot extract, even though the content might be reasonable.

**Why it would happen:** Training changes the model's output distribution. If training pushes toward longer outputs or different formatting, the `extract_json_from_text()` regex patterns might fail.

**How to detect:** Compare format compliance rate before/after training. If it drops, parser failures are occurring.

**Whether training improved it:** NOT MEASURED

### Category 7: Correct Intent but Suboptimal Action for Minority Cases

**What happens:** Model correctly identifies damaged_screen/claim_routing intent but still picks ASK_CLARIFICATION instead of ROUTE_CLAIM.

**Why it would happen:** ROUTE_CLAIM is ideal in only 15% of train but 1% of test. Model has limited exposure. ASK_CLARIFICATION is the safe default with highest expected reward.

**How to detect:** Per-action accuracy on ROUTE_CLAIM and PROVIDE_STEP specifically.

**Whether training improved it:** NOT MEASURED

---

## Generalization Assessment

### Evidence of Memorization vs Generalization: INSUFFICIENT EVIDENCE

No experiments have been run. Cannot assess.

### Recommended Generalization Tests

| Test | Description | Why It Matters |
|------|------------|----------------|
| Paraphrased messages | Rewrite 10 test messages with different wording | Tests beyond template patterns |
| Multi-issue messages | "My screen is cracked AND it won't charge" | Tests compositional understanding |
| Irrelevant information | Add unrelated details to customer messages | Tests noise robustness |
| Angry/frustrated tone | "I'm so frustrated! Nothing works!" | Tests tone robustness |
| Contradictory information | "Screen is black but I can see a crack" | Tests reasoning |
| Already-tried steps | "I already restarted twice" | Tests whether model adapts action |
| Unseen device names | Use devices not in DEVICES list | Tests device-independence |

### Expected Verdict After Experiments

Given the limited data (314 scenarios, 189 unique templates) and narrow evaluation (same template style):

**Most likely verdict: INSUFFICIENT EVIDENCE to distinguish memorization from generalization.**

To demonstrate generalization convincingly, the project would need:
1. A held-out set with genuinely novel phrasings (not template variants)
2. Performance on novel scenarios within 90% of template-based performance
3. Behavioral examples showing the model reasoning about novel situations

---

## Behavioral Change Evidence Framework

When experiments are run, the most convincing evidence is not:
> "Composite reward increased from 0.45 to 0.52"

But rather:
> "On scenario X, the baseline model chose ASK_CLARIFICATION (score 0.5) but the post-trained model chose SEARCH_KB (score 1.0) because the customer had already provided enough information."

**Collect 3-5 concrete behavior-change examples** where:
1. Baseline chooses wrong/suboptimal action
2. Post-trained chooses correct action
3. The reason is explainable (not just random)

Also collect examples where post-training made things WORSE:
1. Baseline was correct
2. Post-trained is wrong
3. Explain why (reward hacking? distribution shift?)

These paired examples are more convincing than any aggregate metric.
