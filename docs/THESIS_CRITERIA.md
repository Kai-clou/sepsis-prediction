# Honours Thesis - Assessment Criteria

## Intent
Demonstrate competency with a particular research skill. Apply critical and analytical skills in the process of completing a research project aimed at contribution to discipline knowledge. Apply relevant methods and/or evaluation tools and frameworks to address discipline specific problems.

## Task
A research report or research paper outlining:
- Motivation behind the topic choice
- Short literature review (summary of work done in Technology Research Preparation)
- Detailed description of the research undertaken
- Methodology used (summary of work done in Technology Research Methodologies)
- Results and conclusions
- Outline of future research in the area

**Due:** Week 14 - Can be negotiated with supervisor

---

## Evaluation Criteria

### (1) Description of model/experiment + knowledge of related work (20%)
- How clearly the model or experiment is presented
- How easy the thesis is to understand
- Logical development and presentation
- Good structure and readability, mostly follows the recommended layout
- For a computer model: description of how the model works
- For an empirical study: description of how the study answers the question
- Description of the results

### (2) Creative Work + Descriptive, Creative, Analytical, Persuasive and Critical Skills (80%)
- Significance, originality, or contribution to the field
- How well has the task been done? (amount of work, elegance, correctness, testing/evaluation, generality)
- For a computer system: elegance/clarity of code, reliability/power of system, ability to generalise
- For an empirical study: design/execution of study, experimental probes/surveys, correctness of analysis, ability to generalise
- Workload

---

## Requirements

### Volume
- Thesis should reflect ~300 hours of work (12cp subject)
- Minimum 20 pages, not including references or appendices
- Initial chapters (Abstract, Introduction) from Technology Research Methods — won't count in TurnItIn

### Required Layout
1. Title Page (name, student number, title, supervisor name)
2. Abstract (approx 300 words)
3. Table of Contents
4. Introduction (Background, Research Significance, Research Questions, Literature Review)
5. Methodology
6. Results
7. Discussion
8. Future Work
9. Conclusion
10. References
11. Appendices
12. Attachments (removed if uploaded to library)
13. Ethics Approval (if applicable)
14. Declaration (assignment coversheet)

**HINT:** Acknowledge individuals (academic advisor, parents, significant others) and organisations which supported your thesis work.

---

## Rubric (Total: 70 pts)

| Criteria | Excellent | Good | Pass | Pts |
|----------|-----------|------|------|-----|
| **Abstract** — Succinct and clear | 5–4 | 4–2 (minor issues or length) | 2–0 | /5 |
| **Written work** — Structure, readability, recommended layout, clear APA7 citations, 20-30+ pages | 10–7 (professionally written) | 7–3 (minor issues, one missing element) | 3–0 (too short, poor structure) | /10 |
| **Body** — Intro includes lit review + RQs, easy to understand, logical flow | 5–3 (clear logical flow from RQ→method→results) | 3–2 (minor flaws in logic) | 2–0 (major issues) | /5 |
| **Analysis** — Analyses results addressing RQs, conclusion recommends future work | 10–7 (clearly links to RQ, future work shows research degree potential) | 7–2 (minor flaws/omissions) | 2–0 (poor/missing analysis) | /10 |
| **Significance** — Originality or contribution to field | 10–7 (publishable quality) | 7–3 (could be mid-level conference poster) | 3–0 (not Honours quality) | /10 |
| **Task** — Amount of work, elegance, correctness, testing, generality | 20–14 (superior work, discipline standard PM, valid results, generalisable) | 14–6 (credit-pass effort, some flaws) | 6–0 (barely done) | /20 |
| **Artefacts** — Systems, products, prototypes professionally developed | 10–7 (superior artefacts demonstrated) | 7–3 (met expectations, minor issues) | 3–0 (poor quality) | /10 |

**Moderation:** Results in HD range (60+) require a 2nd academic marker.

---

## How Our Thesis Maps to the Rubric

| Criteria | Our Coverage | Status |
|----------|-------------|--------|
| **Abstract** (5 pts) | ~300 words, covers motivation, method, 4 key findings, results | Done |
| **Written work** (10 pts) | Full structure following recommended layout, APA7 refs | Done |
| **Body** (5 pts) | Lit review (1.4), 4 RQs (1.3), logical flow from RQ→method→7 experiments→results | Done |
| **Analysis** (10 pts) | 7 experiments analysed, 4 RQs addressed in Discussion, future work section | Done |
| **Significance** (10 pts) | Novel multi-agent architecture, patient-level AUROC 0.8565 beats clinical scores, baseline comparison | Done |
| **Task** (20 pts) | 7 experiments, 65K patient dataset, baseline comparisons, patient-level eval, full ML pipeline | Done |
| **Artefacts** (10 pts) | GitHub repo, training notebooks, metrics notebook, baseline notebook, saved models | Done |

### Key Strengths for Assessment
- **Systematic experimentation:** 7 versions with controlled variable changes
- **Full data pipeline:** Raw MIMIC-IV → preprocessing → labelling → training → evaluation
- **Multiple evaluation levels:** Sequence-level, patient-level, baseline comparison, clinical score comparison
- **Patient-level AUROC 0.8565** — strong result that beats clinical scores and ML baselines
- **Honest limitations:** Near-uniform agent weights, retrospective labelling, single-centre data
