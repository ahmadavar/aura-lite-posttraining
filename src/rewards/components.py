"""Individual reward components -- each is deterministic except
R_clarification which uses keyword matching (no LLM judge needed
for MVP)."""

from typing import List, Set

from src.environment.actions import ModelOutput, VALID_ACTIONS
from src.environment.state import GroundTruth


def r_intent(output: ModelOutput, gt: GroundTruth) -> float:
    """Exact match of predicted intent vs ground truth.
    Returns: 0.0 or 1.0"""
    predicted = output.intent_assessment.strip().lower()
    expected = gt.true_intent.strip().lower()
    return 1.0 if predicted == expected else 0.0


def r_action(output: ModelOutput, gt: GroundTruth) -> float:
    """Action type correctness.
    Exact match to ideal = 1.0
    Match to acceptable = 0.5
    Wrong = 0.0"""
    action = output.action
    if action == gt.ideal_action:
        return 1.0
    if action in gt.acceptable_actions:
        return 0.5
    return 0.0


def r_missing_info(output: ModelOutput, gt: GroundTruth) -> float:
    """Recall of identified missing information gaps.
    Score = |predicted & required| / |required|
    Penalize false positives: -0.1 per FP (floor at 0)."""
    if not gt.required_information:
        # No info required -- if model listed none, perfect
        return 1.0 if not output.missing_information else 0.5

    required: Set[str] = {
        r.strip().lower() for r in gt.required_information
    }
    predicted: Set[str] = {
        p.strip().lower() for p in output.missing_information
    }

    # Fuzzy matching: check if any predicted item is a substring
    # of a required item or vice versa
    true_positives = 0
    matched_required = set()
    for pred in predicted:
        for req in required:
            if pred in req or req in pred:
                if req not in matched_required:
                    true_positives += 1
                    matched_required.add(req)
                    break

    recall = true_positives / len(required)
    false_positives = len(predicted) - true_positives
    penalty = false_positives * 0.1

    return max(0.0, recall - penalty)


def r_clarification(
    output: ModelOutput, gt: GroundTruth
) -> float:
    """Relevance of clarification question to acceptable topics.
    Uses keyword matching against acceptable_clarification_topics.
    Returns N/A sentinel (-1.0) if action is not ASK_CLARIFICATION."""
    if output.action != "ASK_CLARIFICATION":
        return -1.0  # N/A sentinel

    if not gt.acceptable_clarification_topics:
        return 0.5  # No topics defined, neutral score

    question = output.action_detail.strip().lower()
    topics = [t.strip().lower() for t in gt.acceptable_clarification_topics]

    # Count how many acceptable topics are mentioned in the question
    hits = sum(1 for topic in topics if topic in question)

    # Also check individual words from topics
    topic_words = set()
    for t in topics:
        topic_words.update(t.split("_"))
        topic_words.update(t.split())
    topic_words.discard("")

    question_words = set(question.split())
    word_overlap = len(topic_words & question_words)

    if hits > 0:
        return min(1.0, hits / len(topics))

    if word_overlap >= 2:
        return min(1.0, word_overlap / (len(topic_words) + 1) * 2)

    return 0.0


def r_efficiency(output: ModelOutput, gt: GroundTruth) -> float:
    """Penalizes unnecessary clarification when enough info exists.
    If model asks clarification but no info is actually missing: 0.0
    Otherwise: 1.0"""
    if output.action != "ASK_CLARIFICATION":
        return 1.0

    if not gt.required_information:
        # Nothing required but model asked anyway
        return 0.0

    return 1.0


def r_should_not(output: ModelOutput, gt: GroundTruth) -> float:
    """Check model did NOT choose a forbidden action.
    Returns: 0.0 if action is in should_not_do, else 1.0"""
    if output.action in gt.should_not_do:
        return 0.0
    return 1.0


def r_format(output: ModelOutput) -> float:
    """Valid JSON with all required fields.
    Returns: 0.0 or 1.0"""
    return 1.0 if output.is_valid else 0.0
