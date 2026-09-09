# DESIGN SYSTEM — Aura Lite Presentation

---

## 3-COLOR PALETTE

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| **Primary (Navy)** | Dark navy | #1B2A4A | Titles, structure, baselines, neutral technical content, card borders |
| **Positive (Teal)** | Teal green | #0D9488 | GRPO results, improvements, positive deltas, trained model highlights |
| **Warning (Orange)** | Warm orange | #E8713A | Failures, regressions, limitations, degenerate policies |

### Neutral Palette
| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Background | White | #FFFFFF | Slide background |
| Card background | Light gray | #F5F6F8 | Card fills, alternating table rows |
| Body text | Dark gray | #333333 | All body copy |
| Muted text | Medium gray | #888888 | Footnotes, secondary labels |
| Divider lines | Light border | #DDDDDD | Table borders, section dividers |

---

## TYPOGRAPHY

| Element | Font | Size | Weight | Color |
|---------|------|------|--------|-------|
| Slide title | Helvetica | 28pt | Bold | #1B2A4A |
| Slide subtitle | Helvetica | 16pt | Regular | #1B2A4A |
| KPI number (large) | Helvetica | 48pt | Bold | #0D9488 or #1B2A4A |
| KPI delta | Helvetica | 24pt | Bold | #0D9488 (positive) or #E8713A (negative) |
| KPI label | Helvetica | 12pt | Regular | #888888 |
| Card title | Helvetica | 14pt | Bold | #1B2A4A |
| Card body | Helvetica | 12pt | Regular | #333333 |
| Body text | Helvetica | 13pt | Regular | #333333 |
| Chart label | Helvetica | 11pt | Regular | #333333 |
| Footnote | Helvetica | 10pt | Regular | #888888 |
| Takeaway sentence | Helvetica | 14pt | Italic | #333333 |

---

## CARD STYLE

- Background: #F5F6F8
- Border: 1pt solid #DDDDDD
- Corner radius: 8pt
- Padding: 16pt internal
- Shadow: none
- Max cards per row: 4
- Cards within a row: equal width, equal height

---

## CHART STYLE

- Bar fill (positive): #0D9488 (teal)
- Bar fill (negative): #E8713A (orange)
- Bar fill (neutral/baseline): #1B2A4A (navy)
- Bar fill (degenerate): #CCCCCC (gray)
- Reference line: 1.5pt dashed #1B2A4A
- Axis lines: 0.5pt solid #DDDDDD
- Direct-label all bars (no separate legend)
- No gridlines unless chart requires them
- No 3D effects, no gradients, no shadows
- Value labels: 11pt Helvetica, right-aligned on bars

---

## TABLE STYLE

- Header row: #1B2A4A background, white text, Helvetica Bold 11pt
- Body rows: alternating white / #F5F6F8
- Cell text: Helvetica 11pt, #333333
- Border: 0.5pt solid #DDDDDD
- Cell padding: 6pt vertical, 8pt horizontal
- Alignment: numbers center, text left
- No heavy borders, no colored cell fills beyond alternating rows

---

## LAYOUT RULES

- Margins: 0.75" all sides
- Title: top-left aligned, 28pt, consistent position across all slides
- Subtitle/takeaway: directly below title, 1 line max
- Content area: starts below subtitle, generous whitespace
- KPI cards: arranged in row of 4, centered
- Charts: centered, max 85% of content width
- Footer: author + date, bottom-center, 10pt muted gray
- One major visual per slide (chart OR card grid OR pipeline, not multiple)

---

## SPACING

- Title to subtitle: 8pt
- Subtitle to content: 24pt
- Between cards in a row: 12pt
- Between chart and annotation: 16pt
- Between sections on same slide: 20pt
- Bottom takeaway to footer: 24pt minimum

---

## ICON STYLE

- Use simple geometric shapes (circles, rounded rectangles) as action block indicators
- No detailed illustrations or clip art
- Action labels inside rounded rectangles, not icon-dependent
- Pipeline arrows: simple straight or right-angle, 2pt stroke, navy color

---

## PIPELINE / FLOW STYLE

- Direction: top-to-bottom or left-to-right (consistent within slide)
- Boxes: rounded rectangle, #F5F6F8 fill, #1B2A4A border
- Arrows: 2pt solid #1B2A4A
- Label inside box: Helvetica Bold 12pt
- Side channels (e.g., ground truth): dashed line, lighter gray

---

## SLIDE BACKGROUND

- All slides: solid white (#FFFFFF)
- No gradient backgrounds
- No decorative elements
- No watermarks
