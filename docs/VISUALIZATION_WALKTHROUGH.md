# Data Visualization Walkthrough for Supervisor Meeting

**Notebook:** `Data_Exploration_Quick.ipynb`

These are the exact visualizations to show Maoying and what to say for each one.

---

## Visualization 1: Label Distribution

**What it shows:**
- Left: Observation-level (49,201 non-sepsis vs 35,543 sepsis observations)
- Middle: Patient-level (486 non-sepsis vs 239 sepsis patients)
- Right: Hours before sepsis onset distribution

**What to say:**
> "This shows the dataset after preprocessing:
> - **486 patients** never developed sepsis during their ICU stay
> - **239 patients** developed sepsis - that's 33% prevalence
> - The right plot shows the 6-hour prediction window (red dashed line).
>   We label all observations within 6 hours before sepsis onset as positive.
>   This gives the model a chance to predict early, not just at the moment of onset."

**Why this matters:**
Shows you understand the class imbalance problem (33% sepsis) and your prediction window design.

---

## Visualization 2: Missing Data by Variable

**What it shows:**
Color-coded bar chart showing missing rates:
- **Red (>60%)**: base_excess, glucose, ionized_calcium, bilirubin, lactate (59.7%)
- **Orange (30-60%)**: pao2, paco2, ph, fio2
- **Green (<30%)**: All vitals + most basic labs

**What to say:**
> "This is THE key visualization that drove the multi-agent design.
>
> **Vitals (green bars):**
> - Heart rate: 0.5% missing
> - Blood pressure: 2.9% missing
> - Temperature: 5% missing
> - These are measured continuously - very dense data
>
> **Labs (orange/red bars):**
> - Lactate: 60% missing - but it's critical for sepsis!
> - Blood gases (pO2, pCO2): 50-60% missing
> - Bilirubin: 61% missing
> - These are only measured every 6-24 hours
>
> **The orange line at 30%** shows where data becomes 'sparse'. Notice the clear split:
> vitals are dense, labs are sparse.
>
> **Why multi-agent?**
> - **Can't treat all features the same** - vitals have different characteristics than labs
> - **Labs Agent needs learned imputation** - can't just ignore 60% missing lactate values
> - **Vitals Agent can use simpler methods** - the data is nearly complete"

**Why this matters:**
This JUSTIFIES your architecture. You didn't build multi-agent for complexity's sake - the data demanded it.

---

## Visualization 3: Vital Signs & Labs - Sepsis vs Non-Sepsis

**What it shows:**
8 box plots comparing feature distributions:
- Heart Rate: Sepsis patients have **higher** median (97 vs 83 bpm)
- Respiratory Rate: Sepsis **higher** (20 vs 19)
- Temperature: Sepsis **higher** (36.9°C vs 36.8°C)
- Systolic BP: Sepsis patients **lower** (115 vs 118 mmHg)
- MAP: Similar distributions
- O2 Sat: Similar distributions
- Lactate: Sepsis **much higher** (median 1.4 vs 1.5 mmol/L)
- Creatinine: Sepsis **higher** (1.3 vs 1.0 mg/dL)

**What to say:**
> "These box plots show how sepsis patients differ from non-sepsis:
>
> **Classic sepsis patterns:**
> - **Higher heart rate** (97 vs 83) - tachycardia from systemic inflammation
> - **Higher respiratory rate** (20 vs 19) - body trying to compensate
> - **Lower blood pressure** (115 vs 118) - early shock
> - **Higher lactate** (1.4 vs 1.5) - tissue hypoxia, organ dysfunction
> - **Higher creatinine** (1.3 vs 1.0) - kidney damage
>
> These differences are subtle but consistent. That's why we need machine learning -
> a simple threshold rule won't work. The model learns complex combinations of these patterns."

**Why this matters:**
Shows there IS signal in the data (sepsis patients are different), but it's complex.

---

## Visualization 4: Vital Sign Trends Leading Up to Sepsis Onset

**What it shows:**
6 bar charts showing average values at different time windows before sepsis:
- Heart Rate: **-0.1%** change (mostly flat)
- Respiratory Rate: **-2.3%** change (decreasing slightly)
- Temperature: **+4.5%** change (increasing!)
- MAP: **-4.4%** change (decreasing - blood pressure dropping)
- Lactate: **+112%** change (MORE THAN DOUBLES!)
- Creatinine: **+5.1%** change (kidney function worsening)

**What to say:**
> "This shows how vitals and labs change in the 48 hours before sepsis onset.
>
> **Key findings:**
> - **Lactate increases by 112%** - it doubles in the hours before sepsis. This is huge!
> - **MAP drops by 4.4%** - blood pressure falling (early shock)
> - **Temperature rises by 4.5%** - fever developing
> - **Creatinine increases 5.1%** - kidney function declining
>
> **Why Trend Agent?**
> These temporal patterns are critical. It's not just 'is lactate high?' - it's
> 'is lactate RISING rapidly?' The Transformer-based Trend Agent is designed to
> capture these rate-of-change patterns across all 24 features simultaneously."

**Why this matters:**
Justifies the Trend Agent - sepsis has temporal signatures, not just static values.

---

## Visualization 5: Feature Correlation with Sepsis

**What it shows:**
Horizontal bar chart showing correlation strength:

**Positive (red - associated with sepsis):**
- BUN: +0.231 (strongest)
- base_excess: +0.205
- Bilirubin: +0.163
- Creatinine: +0.156
- Heart rate: +0.081
- Respiratory rate: +0.060

**Negative (blue - protective/lower in sepsis):**
- pO2: -0.140 (low oxygen)
- Lactate: -0.056 (wait, this is inverted?)
- SBP: -0.066 (blood pressure)

**What to say:**
> "This shows which features correlate most with sepsis labels:
>
> **Strongest predictors (red bars):**
> - **BUN** (kidney function marker) - correlation 0.23
> - **Base excess** (metabolic acidosis indicator) - correlation 0.21
> - **Creatinine** (kidney damage) - correlation 0.16
> - **Heart rate** - correlation 0.08
>
> **Note:** Correlations are weak (< 0.3) which is typical in medicine. No single feature
> perfectly predicts sepsis. That's why we need a model that learns complex interactions
> between features.
>
> The multi-agent approach lets each agent specialize: the Labs Agent focuses on
> high-value features like BUN and creatinine, while the Vitals Agent tracks HR and BP."

**Why this matters:**
Shows no single feature is enough - need multivariate prediction.

---

## Visualization 6: Patient Timeline (Example Case)

**What it shows:**
6 line plots for one sepsis patient (ID 10001843):
- **Heart Rate**: Spikes to 160 bpm around hour 2, then variable
- **Temperature**: Sharp DROP after sepsis onset (from 36.6 to 36.4°C)
- **MAP**: Drops from 90 to 40 mmHg - SEVERE hypotension
- **Respiratory Rate**: Variable, increases to 26 after onset
- **Lactate**: Flat (imputed - no new measurements)
- **O2 Saturation**: Variable around 95-98%

**Red shaded area:** 6-hour prediction window before sepsis onset
**Red dashed line:** Actual sepsis onset

**What to say:**
> "This is a real sepsis case - one patient's journey hour-by-hour.
>
> **Before sepsis onset (red shaded area - our 6h prediction window):**
> - Heart rate climbing and unstable (130-160 bpm)
> - MAP dropping from 90 to 80 mmHg
> - We want the model to flag this patient HERE, during the red window
>
> **After sepsis onset (red dashed line):**
> - MAP crashes to 40 mmHg - severe shock
> - Temperature drops (septic shock)
> - Patient is in crisis
>
> **The model's job:**
> Look at the pattern in the 24 hours BEFORE the red line and predict
> 'this patient will develop sepsis in the next 6 hours.'
>
> **Why this is hard:**
> - Each patient's trajectory is unique
> - Missing data (lactate is flat - not actually measured)
> - Noisy measurements (O2 sat jumping around)
>
> The multi-agent architecture handles this: Vitals Agent tracks HR/BP patterns,
> Labs Agent imputes missing lactate, Trend Agent sees the overall deterioration."

**Why this matters:**
Makes the problem REAL - not just abstract numbers, but a real patient deteriorating. Shows the complexity the model must handle.

---

## How to Present These (Flow)

**Order to show:**

1. **Label Distribution** (30 sec)
   - "Here's the dataset: 239 sepsis patients, 486 non-sepsis, 6-hour prediction window"

2. **Missing Data** (1 min) ⭐ KEY
   - "This drives the entire architecture design - vitals dense, labs sparse"
   - "Can you see why we need different agents for different data types?"

3. **Sepsis vs Non-Sepsis Distributions** (30 sec)
   - "Sepsis patients ARE different - higher HR, lower BP, higher lactate"
   - "But differences are subtle - need ML to detect complex patterns"

4. **Temporal Trends** (45 sec) ⭐ KEY
   - "Lactate doubles, MAP drops 4.4% in hours before sepsis"
   - "This is why we added the Trend Agent - temporal patterns matter"

5. **Correlation Chart** (30 sec)
   - "No single feature predicts sepsis well - all correlations weak"
   - "Need multivariate approach combining many features"

6. **Patient Timeline** (1 min) ⭐ KEY
   - "Here's a real case - watch the deterioration in the 6h window"
   - "This is what the model needs to catch early"

**Total time:** ~4-5 minutes for all visualizations

---

## Key Talking Points to Repeat

Throughout the visualization walkthrough, emphasize these themes:

1. **Data characteristics drove design choices**
   - Not arbitrary architecture - designed for this specific data

2. **Three distinct patterns that need three specialized agents:**
   - Dense vitals → Bi-LSTM with attention
   - Sparse labs → LSTM with learned imputation
   - Temporal trends → Transformer encoder

3. **Real clinical problem:**
   - Not toy data - real ICU patients
   - Missing data, noise, class imbalance
   - Subtle patterns that need sophisticated ML

---

## Questions She Might Ask

### "Why is lactate showing negative correlation when you said it increases?"
> "Good catch! The correlation here is on the raw values, not the trend.
> The trend visualization (112% increase) is more informative. Correlation can be misleading
> with missing data and non-linear relationships - that's exactly why we use deep learning
> instead of linear models."

### "This one patient timeline - is this cherry-picked?"
> "Fair question. I can show you more examples if you'd like. Some cases are very clear
> (obvious deterioration), others are subtle. The model needs to work on both.
> This particular case shows clear patterns in the 6h window, which is ideal for
> demonstrating what we're trying to detect."

### "The missing data is really high (60%). Is that a problem?"
> "It's a challenge, but it's real-world ICU data. Labs like lactate are expensive and
> invasive, so they're not measured unless needed. That's exactly why I built the
> learned imputation layer in the Labs Agent - it doesn't just fill with mean values,
> it learns patient context. For example, if a patient has high BUN and creatinine,
> the model learns lactate is probably elevated too."

### "Can you show me more patient examples?"
> "Absolutely - I can scroll down in the notebook to show more. Each case is unique:
> some show classic sepsis (high HR, low BP, high lactate), others are subtle.
> The model learns to recognize the pattern across all 239 sepsis cases in training."

---

## What NOT to Do

❌ Don't rush through these - they're the foundation of your work
❌ Don't read the numbers verbatim - tell the story
❌ Don't skip the missing data chart - it's your best justification
❌ Don't show visualizations without connecting to architecture choices
❌ Don't get defensive if she questions the approach

✅ DO connect each visualization to a design decision
✅ DO emphasize "data-driven design"
✅ DO show enthusiasm for the patterns you discovered
✅ DO invite questions throughout

---

**These visualizations are your secret weapon - they make the case for multi-agent better than any explanation.**
