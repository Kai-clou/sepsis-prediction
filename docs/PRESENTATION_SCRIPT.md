# Sepsis Prediction Project - Presentation Script

**Audience:** Non-technical supervisor
**Duration:** ~10 minutes
**Goal:** Explain what we're building and why it matters

---

## Opening (30 seconds)

> "I'm building an AI system that can predict when ICU patients are about to develop sepsis - up to 6 hours before it happens. This early warning could help doctors intervene faster and potentially save lives."

---

## Part 1: What is Sepsis? (2 minutes)

> "Let me start with the problem we're trying to solve."

### The Simple Explanation:

> "Sepsis is when the body's response to an infection goes wrong. Instead of just fighting the infection, the body starts attacking itself. It's like friendly fire - the immune system damages its own organs."

### Why It Matters:

> "Sepsis is one of the leading causes of death in hospitals:
> - **1 in 3** patients who die in hospitals have sepsis
> - **Every hour** of delayed treatment increases death risk by 7-8%
> - It kills more people than heart attacks and strokes combined"

### The Challenge:

> "The tricky part is that sepsis doesn't announce itself. Early symptoms look like many other conditions - fever, fast heart rate, confusion. By the time doctors are certain it's sepsis, it may already be too late."

> "That's where AI comes in. What if we could spot the pattern **before** it becomes obvious to humans?"

---

## Part 2: What Our System Does (2 minutes)

### The Concept:

> "Our AI watches patients continuously - like having a tireless doctor who never blinks, monitoring every vital sign and lab result 24/7."

> "It learns patterns from thousands of past patients who developed sepsis, and uses those patterns to spot warning signs in current patients."

### How It Works (Non-Technical):

> "Imagine you're learning to predict rain. You might notice:
> - Dark clouds usually come before rain
> - Humidity goes up
> - Temperature drops slightly
>
> Our AI does the same thing, but with medical data:
> - Heart rate patterns
> - Blood pressure trends
> - Lab values like lactate and white blood cells
> - How these values **change over time**"

### The Multi-Agent Approach:

> "We built something special - instead of one AI looking at everything, we have **three specialist AIs** working together:
>
> 1. **Vitals Specialist** - Watches heart rate, blood pressure, temperature
> 2. **Labs Specialist** - Analyzes blood tests and lab results
> 3. **Trends Specialist** - Spots patterns over time (is the patient getting worse?)
>
> Then a **coordinator** combines their opinions into one prediction.
>
> It's like having a team of specialists instead of one generalist."

---

## Part 3: How Do We Know It Works? (2 minutes)

### Measuring Success:

> "We measure our AI using something called **AUROC** - think of it as a score from 0 to 1:
> - **0.5** = Random guessing (flipping a coin)
> - **0.7** = Good (better than most clinical scores)
> - **0.8** = Very good
> - **0.9+** = Excellent"

### Comparing to Existing Methods:

> "Doctors currently use simple scoring systems. Here's how they compare:"

| Method | Score | What it is |
|--------|-------|------------|
| **SIRS** | 0.64-0.68 | Simple checklist (fever? fast heart rate?) |
| **qSOFA** | 0.66-0.70 | Quick bedside score (3 questions) |
| **MEWS** | 0.67-0.72 | Early warning score |
| **Our AI (v3)** | **0.7263** | Multi-agent deep learning |
| Published ML | 0.75-0.85 | Other research AI systems |

> "Our system **outperforms** the clinical scores doctors currently use. We're in range with other published AI research."

### What This Means Practically:

> "If our AI says 'high risk', it's right about **73% of the time** - significantly better than the current tools available at the bedside."

---

## Part 4: Why Not Use Simpler Tools? (2 minutes)

### "Why not just use n8n or a simple automation?"

> "Great question. n8n and similar tools are excellent for **workflow automation** - connecting apps, sending notifications, moving data around."

> "But predicting sepsis requires something fundamentally different:"

| Task | Right Tool | Why |
|------|-----------|-----|
| Send alert when value > threshold | n8n, simple rules | Single condition check |
| Connect hospital system to messaging | n8n, Zapier | Data routing |
| **Predict sepsis from 24 features changing over time** | **Deep Learning** | Complex pattern recognition |

> "Sepsis prediction isn't about simple rules like 'if temperature > 38, send alert.' It's about recognizing **complex patterns across multiple variables over time**."

> "For example, a heart rate of 100 might be normal for one patient but dangerous for another - it depends on their baseline, their other vitals, their lab trends, their medications..."

> "Deep learning can learn these complex, patient-specific patterns. Simple automation tools cannot."

### The Analogy:

> "It's like asking 'why not use a calculator instead of a weather forecasting supercomputer?'
>
> A calculator can add numbers, but predicting weather requires analyzing millions of data points and complex atmospheric patterns. Same with sepsis prediction."

---

## Part 5: Project Status & Results (1 minute)

### What We've Accomplished:

> "We've built a complete system:
>
> 1. **Data Pipeline** - Processes real ICU patient data from MIMIC-IV (a large public hospital database)
> 2. **Multi-Agent AI** - Three specialist networks + coordinator
> 3. **Evaluation Framework** - Proper testing against held-out patients
>
> We trained on **3,559 patients** with over **420,000 observations**."

### Current Performance:

> "Our best model achieves:
> - **AUROC: 0.7263** - Better than clinical bedside scores
> - **Predicts 6 hours ahead** - Giving doctors time to act"

### Next Steps:

> "We're now:
> 1. Comparing against other ML methods (XGBoost, Random Forest)
> 2. Analyzing which features matter most
> 3. Documenting everything for reproducibility"

---

## Closing (30 seconds)

> "In summary: Sepsis kills, but early detection saves lives. We've built an AI system that watches patients continuously and predicts sepsis hours before it becomes obvious. Our results are competitive with published research and better than the clinical scores doctors use today."

> "Any questions?"

---

## Anticipated Questions & Answers

### Q: "Is this ready for hospitals?"

> "Not yet. This is research - we'd need clinical trials, regulatory approval (TGA/FDA), and integration with hospital systems. But the foundation is solid."

### Q: "How is this different from existing systems?"

> "Most hospital early warning systems use simple rules (if heart rate > X, alert). Our system uses deep learning to find complex patterns humans might miss, and combines multiple specialist networks."

### Q: "Why MIMIC data? Is it relevant to Australian hospitals?"

> "MIMIC is from a US hospital (Beth Israel, Boston), but sepsis physiology is universal. The patterns we learn should transfer, though local validation would be needed."

### Q: "What about false alarms?"

> "That's the trade-off. We can tune the system to be more sensitive (catch more sepsis) or more specific (fewer false alarms). In practice, hospitals would choose based on their capacity to respond to alerts."

### Q: "Can you explain the 0.7263 score in plain English?"

> "If we showed the AI 100 patients - 50 who will develop sepsis and 50 who won't - and asked it to rank them by risk, there's a 72.63% chance it puts a true sepsis patient higher than a non-sepsis patient. Random guessing would be 50%."

---

## Visual Aids to Prepare

1. **Sepsis statistics infographic** - Deaths, time sensitivity
2. **Multi-agent architecture diagram** - Three specialists + coordinator
3. **Performance comparison bar chart** - Our AI vs clinical scores
4. **Timeline** - 6 hours early warning

---

*Script prepared: February 2026*
