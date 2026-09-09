#!/usr/bin/env python3
"""Generate the Aura Lite GRPO presentation PDF."""

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "docs", "PRESENTATION.pdf")

# Colors
DARK = HexColor("#1a1a2e")
ACCENT = HexColor("#0f3460")
HIGHLIGHT = HexColor("#e94560")
LIGHT_BG = HexColor("#f0f0f5")
GREEN = HexColor("#2d6a4f")
ORANGE = HexColor("#e76f51")

styles = getSampleStyleSheet()

# Custom styles
styles.add(ParagraphStyle(
    "SlideTitle", parent=styles["Heading1"],
    fontSize=28, leading=34, textColor=DARK,
    spaceAfter=20, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "SlideSubtitle", parent=styles["Normal"],
    fontSize=16, leading=20, textColor=ACCENT,
    spaceAfter=12, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "SectionHead", parent=styles["Heading2"],
    fontSize=22, leading=28, textColor=DARK,
    spaceBefore=10, spaceAfter=14,
))
styles.add(ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=13, leading=18, textColor=black,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    "BodyBold", parent=styles["Normal"],
    fontSize=13, leading=18, textColor=black,
    spaceAfter=8, fontName="Helvetica-Bold",
))
styles.add(ParagraphStyle(
    "BulletItem", parent=styles["Normal"],
    fontSize=13, leading=18, textColor=black,
    spaceAfter=6, leftIndent=20, bulletIndent=8,
    bulletFontName="Helvetica", bulletFontSize=13,
))
styles.add(ParagraphStyle(
    "Verdict", parent=styles["Normal"],
    fontSize=15, leading=22, textColor=DARK,
    spaceAfter=10, fontName="Helvetica-Bold",
    alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "SmallNote", parent=styles["Normal"],
    fontSize=10, leading=13, textColor=HexColor("#666666"),
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    "PieceNum", parent=styles["Normal"],
    fontSize=16, leading=20, textColor=HIGHLIGHT,
    spaceAfter=4, fontName="Helvetica-Bold",
))


def make_table(data, col_widths=None, highlight_last_row=False):
    """Create a styled table."""
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]
    if highlight_last_row:
        style_cmds.append(("BACKGROUND", (0, -1), (-1, -1), HexColor("#e8f5e9")))
        style_cmds.append(("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"))
    t.setStyle(TableStyle(style_cmds))
    return t


def build():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=LETTER,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    story = []
    W = doc.width

    # ===== SLIDE 1: TITLE =====
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph(
        "Aura Lite", styles["SlideTitle"]
    ))
    story.append(Paragraph(
        "Post-Training a Customer Support<br/>Decision Specialist with GRPO",
        styles["SlideSubtitle"]
    ))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(
        "GRPO reinforcement learning on Qwen2.5-1.5B<br/>"
        "7-component reward function | 266 training scenarios | LoRA adapters",
        styles["Body"]
    ))
    story[-1].style = ParagraphStyle(
        "CenterBody", parent=styles["Body"], alignment=TA_CENTER,
        textColor=HexColor("#555555"),
    )
    story.append(Spacer(1, 0.8 * inch))
    story.append(Paragraph("Ahmad Naggayev | September 2026", styles["SlideSubtitle"]))
    story.append(PageBreak())

    # ===== SLIDE 2: THE PROBLEM =====
    story.append(Paragraph("The Problem", styles["SectionHead"]))
    story.append(Paragraph(
        "A customer messages support about a broken phone. The AI agent must decide "
        "what to do next -- not generate text, but make a <b>triage decision</b>.",
        styles["Body"]
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("6 possible actions:", styles["BodyBold"]))

    action_data = [
        ["Action", "Real-World Meaning"],
        ["ASK_CLARIFICATION", "\"Can you tell me more?\" -- need more info"],
        ["SEARCH_KB", "\"Let me look that up\" -- search knowledge base"],
        ["PROVIDE_STEP", "\"Try this fix\" -- give a troubleshooting step"],
        ["ROUTE_CLAIM", "\"Transferring to claims\" -- insurance/warranty"],
        ["ESCALATE", "\"Getting a supervisor\" -- beyond normal support"],
        ["COMPLETE", "\"Glad I could help\" -- problem is resolved"],
    ]
    story.append(make_table(action_data, col_widths=[W * 0.35, W * 0.65]))
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "<b>The challenge:</b> The base model (Qwen2.5-1.5B) knows customer support language, "
        "but it collapses to one action -- SEARCH_KB ~80% of the time. Even with a carefully "
        "engineered prompt, it lacks a consistent decision policy.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "<b>The question:</b> Can RL post-training teach the model a better policy "
        "than prompt engineering alone?",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ===== SLIDE 3: THE METHOD =====
    story.append(Paragraph("The Method: GRPO", styles["SectionHead"]))
    story.append(Paragraph(
        "<b>GRPO = Group Relative Policy Optimization</b> -- a reinforcement learning method "
        "from the same family as DeepSeek-R1.",
        styles["Body"]
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph("How it works (4 steps, repeated 266 times):", styles["BodyBold"]))
    steps = [
        "Show the model a customer scenario",
        "Model generates 4 different responses (K=4 rollouts)",
        "Score each response with the reward function (0 to 1)",
        "Update model weights: make high-scoring responses more likely, low-scoring less likely",
    ]
    for i, s in enumerate(steps, 1):
        story.append(Paragraph(
            f"<b>Step {i}:</b> {s}", styles["BulletItem"], bulletText="\u2022"
        ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Key components:", styles["BodyBold"]))
    components = [
        "<b>Base model:</b> Qwen2.5-1.5B-Instruct (1.5 billion parameters)",
        "<b>LoRA adapters:</b> Only 4M trainable parameters (0.3% of total) -- base model frozen",
        "<b>KL regularization:</b> Prevents the model from drifting too far from its original behavior",
        "<b>Training data:</b> 266 synthetic customer support scenarios across 6 device domains",
    ]
    for c in components:
        story.append(Paragraph(c, styles["BulletItem"], bulletText="\u2022"))
    story.append(PageBreak())

    # ===== SLIDE 4: REWARD FUNCTION =====
    story.append(Paragraph("The Reward Function", styles["SectionHead"]))
    story.append(Paragraph(
        "The reward function is the \"teacher\" -- it scores every model response on 7 dimensions. "
        "Like a GPA that combines math, English, science into one number.",
        styles["Body"]
    ))
    story.append(Spacer(1, 8))
    reward_data = [
        ["Component", "Weight", "What It Grades"],
        ["Action correctness", "25%", "Did you pick the right action?"],
        ["Intent accuracy", "20%", "Did you diagnose the problem correctly?"],
        ["Missing info recall", "15%", "Did you notice what info was missing?"],
        ["Clarification quality", "15%", "If you asked a question, was it relevant?"],
        ["Efficiency", "10%", "Did you avoid unnecessary questions?"],
        ["Should-not penalty", "10%", "Did you avoid harmful actions?"],
        ["Format compliance", "5%", "Is the output parseable structured JSON?"],
    ]
    story.append(make_table(reward_data, col_widths=[W * 0.28, W * 0.12, W * 0.60]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Why rule-based, not learned?</b> 392 scenarios is too few to train a robust reward model. "
        "Deterministic rules are auditable and interpretable -- you can inspect exactly why "
        "any response got its score.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ===== SLIDE 5: EXPERIMENTAL DESIGN =====
    story.append(Paragraph("Experimental Design", styles["SectionHead"]))
    story.append(Paragraph(
        "4 conditions tested under identical evaluation settings (same parser, metrics, reward weights, temperature=0.3):",
        styles["Body"]
    ))
    story.append(Spacer(1, 8))
    design_data = [
        ["Condition", "Model", "Training", "Prompt"],
        ["Random", "None", "None", "None"],
        ["Zero-shot", "Qwen2.5-1.5B", "None", "Minimal"],
        ["Engineered", "Qwen2.5-1.5B", "None", "Full engineered"],
        ["GRPO (RL)", "Qwen2.5-1.5B + LoRA", "GRPO, 1 epoch", "Full engineered"],
    ]
    story.append(make_table(design_data, col_widths=[W * 0.22, W * 0.30, W * 0.22, W * 0.26],
                            highlight_last_row=True))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Degenerate Policy Gate (pre-training verification):", styles["BodyBold"]))
    story.append(Paragraph(
        "Before training, verified that no trivial strategy scores well:",
        styles["Body"]
    ))
    gate_data = [
        ["Degenerate Policy", "Reward"],
        ["ALWAYS_ASK", "0.500"],
        ["ALWAYS_SEARCH_KB", "0.491"],
        ["ALWAYS_PROVIDE_STEP", "0.455"],
        ["ALWAYS_ESCALATE", "0.435"],
        ["ALWAYS_COMPLETE", "0.365"],
        ["KEYWORD_STUFFER", "0.519"],
    ]
    story.append(make_table(gate_data, col_widths=[W * 0.45, W * 0.25]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "All below 0.55. Any model scoring above 0.616 (engineered baseline) demonstrates "
        "genuine learning, not a shortcut.",
        styles["SmallNote"]
    ))
    story.append(PageBreak())

    # ===== SLIDE 6: MAIN RESULTS =====
    story.append(Paragraph("Results: GRPO Wins on Every Metric", styles["SectionHead"]))
    story.append(Paragraph(
        "83-scenario held-out test set. Model never saw these during training.",
        styles["Body"]
    ))
    story.append(Spacer(1, 8))
    results_data = [
        ["Metric", "Random", "Zero-shot", "Engineered", "GRPO (RL)", "Delta"],
        ["Composite Reward", "0.441", "0.429", "0.616", "0.657", "+0.041"],
        ["Intent Accuracy", "12.0%", "0.0%", "57.8%", "66.3%", "+8.5pp"],
        ["Action Accuracy", "21.7%", "25.3%", "30.1%", "34.9%", "+4.8pp"],
        ["Macro F1", "0.212", "0.150", "0.258", "0.303", "+0.045"],
        ["Format Compliance", "100%", "96.4%", "98.8%", "100%", "+1.2pp"],
        ["Action Entropy", "-", "1.254", "1.012", "1.162", "+0.150"],
    ]
    story.append(make_table(results_data,
                            col_widths=[W * 0.24, W * 0.13, W * 0.15, W * 0.16, W * 0.16, W * 0.16]))
    story.append(Spacer(1, 14))
    story.append(Paragraph("What each metric means:", styles["BodyBold"]))
    metric_explain = [
        "<b>Composite Reward:</b> Overall grade (0-1) combining all 7 reward components",
        "<b>Intent Accuracy:</b> Did the model correctly diagnose the customer's problem?",
        "<b>Action Accuracy:</b> Did the model pick the exact right action out of 6?",
        "<b>Macro F1:</b> Average performance across ALL actions (punishes if model ignores rare actions)",
        "<b>Format Compliance:</b> Were all outputs parseable JSON?",
        "<b>Action Entropy:</b> How diverse are the model's choices? (higher = more variety)",
    ]
    for m in metric_explain:
        story.append(Paragraph(m, styles["BulletItem"], bulletText="\u2022"))
    story.append(PageBreak())

    # ===== SLIDE 7: PER-ACTION BREAKDOWN =====
    story.append(Paragraph("Per-Action Performance", styles["SectionHead"]))
    story.append(Paragraph(
        "Macro F1 is the average of these 6 individual action F1 scores. "
        "Precision = \"when the model makes this call, how often is it right?\" "
        "Recall = \"of all real cases, how many did the model catch?\"",
        styles["Body"]
    ))
    story.append(Spacer(1, 8))
    action_perf_data = [
        ["Action", "Precision", "Recall", "F1", "Support"],
        ["ESCALATE", "0.80", "0.50", "0.62", "8"],
        ["SEARCH_KB", "0.26", "1.00", "0.41", "17"],
        ["ROUTE_CLAIM", "1.00", "0.20", "0.33", "10"],
        ["PROVIDE_STEP", "0.57", "0.21", "0.31", "19"],
        ["ASK_CLARIFICATION", "0.50", "0.09", "0.15", "23"],
        ["COMPLETE", "0.00", "0.00", "0.00", "6"],
    ]
    story.append(make_table(action_perf_data,
                            col_widths=[W * 0.30, W * 0.15, W * 0.15, W * 0.15, W * 0.15]))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Key findings:", styles["BodyBold"]))
    findings = [
        "<b>ESCALATE (F1=0.62):</b> Best performer. When model says \"get a supervisor,\" it's right 80% of the time.",
        "<b>ROUTE_CLAIM (Prec=1.00):</b> Perfect precision -- never sends a non-claim case to claims. "
        "But only catches 2 of 10 real claims (low recall).",
        "<b>SEARCH_KB (Recall=1.00):</b> Catches ALL search cases, but floods with false positives (78% of all predictions). "
        "This is the SEARCH_KB bias that persists even after GRPO.",
        "<b>COMPLETE (F1=0.00):</b> Total failure -- model never predicts \"done.\" Rarest action, "
        "hard to learn from limited data.",
    ]
    for f in findings:
        story.append(Paragraph(f, styles["BulletItem"], bulletText="\u2022"))
    story.append(PageBreak())

    # ===== SLIDE 8: GRPO vs ENGINEERED IMPROVEMENT =====
    story.append(Paragraph("What GRPO Improved", styles["SectionHead"]))
    story.append(Paragraph(
        "Comparing GRPO to the engineered prompt baseline (the best non-RL condition):",
        styles["Body"]
    ))
    story.append(Spacer(1, 8))
    improvement_data = [
        ["Action", "Eng. F1", "GRPO F1", "Change"],
        ["ROUTE_CLAIM", "0.18", "0.33", "Recall doubled (0.10 -> 0.20)"],
        ["PROVIDE_STEP", "0.24", "0.31", "Improved across all metrics"],
        ["ASK_CLARIFICATION", "0.08", "0.15", "Precision improved (0.33 -> 0.50)"],
        ["SEARCH_KB", "0.38", "0.41", "Recall reached 100%"],
        ["ESCALATE", "0.67", "0.62", "Slight decrease (-0.05)"],
        ["COMPLETE", "0.00", "0.00", "No change (both fail)"],
    ]
    story.append(make_table(improvement_data,
                            col_widths=[W * 0.27, W * 0.13, W * 0.13, W * 0.47]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Per-Domain Reward:", styles["BodyBold"]))
    domain_data = [
        ["Domain", "Engineered", "GRPO", "Delta"],
        ["black_screen", "0.818", "0.832", "+0.015"],
        ["liquid_damage", "0.667", "0.789", "+0.123"],
        ["lost_stolen", "0.735", "0.713", "-0.022"],
        ["data_transfer", "0.728", "0.706", "-0.022"],
        ["charging", "0.594", "0.659", "+0.066"],
        ["battery_drain", "0.551", "0.540", "-0.011"],
    ]
    story.append(make_table(domain_data, col_widths=[W * 0.28, W * 0.22, W * 0.22, W * 0.18]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Largest gain on liquid_damage (+0.123) and charging (+0.066). "
        "Minor regression on lost_stolen and data_transfer.",
        styles["SmallNote"]
    ))
    story.append(PageBreak())

    # ===== SLIDE 9: CHALLENGE SET =====
    story.append(Paragraph("Generalization: Challenge Set", styles["SectionHead"]))
    story.append(Paragraph(
        "26 scenarios with novel phrasings the model never saw during training. "
        "Tests whether the model learned generalizable decision-making, not just pattern matching.",
        styles["Body"]
    ))
    story.append(Spacer(1, 8))
    challenge_data = [
        ["Metric", "Test Set (83)", "Challenge Set (26)"],
        ["Composite Reward", "0.657", "0.576"],
        ["Intent Accuracy", "66.3%", "46.2%"],
        ["Action Accuracy", "34.9%", "19.2%"],
        ["Macro F1", "0.303", "0.107"],
        ["Format Compliance", "100%", "100%"],
    ]
    story.append(make_table(challenge_data, col_widths=[W * 0.35, W * 0.30, W * 0.35]))
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Performance drops on novel phrasings (expected). The reward stays above the random baseline (0.441) "
        "and format compliance remains perfect. The drop in Macro F1 (0.303 -> 0.107) shows the SEARCH_KB "
        "bias intensifies on unfamiliar inputs -- the model retreats to its safe default.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "<b>Interpretation:</b> The model learned real decision-making patterns (reward 0.576 > random 0.441), "
        "but generalization to completely novel phrasings needs more diverse training data.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ===== SLIDE 10: TRAINING DETAILS =====
    story.append(Paragraph("Training Details", styles["SectionHead"]))
    training_data = [
        ["Parameter", "Value"],
        ["Method", "GRPO (Group Relative Policy Optimization)"],
        ["Base Model", "Qwen2.5-1.5B-Instruct"],
        ["Trainable Parameters", "4M (LoRA rank 16, alpha 32)"],
        ["Training Scenarios", "266"],
        ["Epochs", "1"],
        ["Rollouts per Prompt (K)", "4"],
        ["Learning Rate", "1e-5"],
        ["KL Beta", "0.1"],
        ["Final Training Reward", "0.787"],
        ["Final KL Divergence", "0.001"],
        ["Training Time", "17.2 hours"],
        ["Hardware", "Apple M1 Pro (MPS backend)"],
    ]
    story.append(make_table(training_data, col_widths=[W * 0.40, W * 0.60]))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Why these choices:", styles["BodyBold"]))
    choices = [
        "<b>GRPO over PPO:</b> No critic network needed. Simpler, cheaper, works with small datasets.",
        "<b>Rule-based reward over learned:</b> 392 scenarios too few for a robust reward model. Rules are auditable.",
        "<b>1.5B model:</b> Feasible on consumer hardware. Forces genuine behavioral learning, not just model scale.",
        "<b>1 epoch:</b> Prevents overfitting on small dataset. Conservative choice -- more epochs could help or hurt.",
        "<b>LoRA:</b> Trains 0.3% of parameters. Base model knowledge preserved, only decision policy changes.",
    ]
    for c in choices:
        story.append(Paragraph(c, styles["BulletItem"], bulletText="\u2022"))
    story.append(PageBreak())

    # ===== SLIDE 11: FINAL VERDICT =====
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Final Verdict", styles["SectionHead"]))
    story.append(Spacer(1, 12))

    verdict_items = [
        ("What GRPO learned:", GREEN, [
            "Better intent diagnosis -- 66.3% vs 57.8% (+8.5pp)",
            "Higher precision on minority actions (ROUTE_CLAIM: 100%, ESCALATE: 80%)",
            "Perfect format compliance -- 100% parseable outputs",
            "Domain-specific improvement -- liquid_damage +0.123, charging +0.066",
        ]),
        ("What GRPO did NOT fully solve:", ORANGE, [
            "SEARCH_KB collapse -- still predicts SEARCH_KB 78% of the time (ground truth: 20%)",
            "COMPLETE action -- never predicted (0/6 scenarios)",
            "Missing information detection -- 0% recall",
            "Generalization to novel phrasings (challenge set reward drops from 0.657 to 0.576)",
        ]),
    ]

    for title, color, items in verdict_items:
        story.append(Paragraph(
            f'<font color="#{color.hexval()[2:]}">{title}</font>',
            styles["BodyBold"]
        ))
        for item in items:
            story.append(Paragraph(item, styles["BulletItem"], bulletText="\u2022"))
        story.append(Spacer(1, 10))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Practice beat instructions.",
        styles["Verdict"]
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "GRPO post-training produced a measurable, consistent improvement over the best "
        "prompt engineering baseline. The model learned better problem diagnosis (intent "
        "accuracy +8.5pp) and more balanced decision-making (macro F1 +0.045). "
        "While the fundamental SEARCH_KB bias persists, this experiment demonstrates that "
        "experience-based learning extracts behavioral signal that instruction-based prompting cannot.",
        styles["Body"]
    ))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Next steps: more training data, more epochs, SEARCH_KB penalty in reward, "
        "multi-turn episodes, real customer messages.",
        styles["SmallNote"]
    ))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "Ahmad Naggayev | ahmadavar956@gmail.com | September 2026",
        ParagraphStyle("Footer", parent=styles["Body"],
                       alignment=TA_CENTER, textColor=HexColor("#888888"), fontSize=11)
    ))

    doc.build(story)
    print(f"Presentation saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
