# The Complete Beginner's Guide to Our Sepsis Prediction Project

**If you know absolutely nothing — start here.** This document walks you through everything, from "what is sepsis" to "how our model works," in plain English.

---

## Part 1: The Medical Problem

### 1.1 What is an ICU?

An **ICU** (Intensive Care Unit) is the most critical ward in a hospital. Patients here are the sickest — they're hooked up to machines that monitor their heart, breathing, blood pressure, oxygen, etc. **24/7**. Nurses and doctors check on them constantly.

These machines generate **a LOT of data** — a heart rate reading every few minutes, blood pressure every few minutes, oxygen every few minutes. That's hundreds of numbers per patient per day.

### 1.2 What is Sepsis?

Your body fights infections using your immune system. Normally, this is a good thing — you get a cut, white blood cells rush in, kill the bacteria, you heal.

**Sepsis** is when this immune response goes haywire. Instead of fighting the infection locally, your body launches an **all-out war** that damages your own organs. It's like calling in a nuclear strike to kill a burglar — yes, the burglar dies, but so does your house.

What happens during sepsis:
1. Patient gets an infection (pneumonia, urinary tract infection, wound infection, etc.)
2. Immune system overreacts
3. Inflammation spreads throughout the entire body
4. Blood pressure drops dangerously (blood can't reach organs)
5. Organs start failing — kidneys, liver, lungs, brain
6. If untreated → **septic shock** → death

### 1.3 Why is Early Detection So Critical?

Here's the terrifying statistic: **every hour you delay treatment, the chance of death goes up by 7.6%.**

- Treated at hour 1: high survival rate
- Treated at hour 6: significantly worse
- Treated at hour 12: possibly too late

The problem? Early sepsis looks like a lot of other things. A patient with a fever and fast heart rate might have sepsis, or might just be anxious, or have a normal post-surgery response. By the time it's **obvious** it's sepsis, organs are already failing.

**Our goal:** Build a computer system that looks at patient data and raises an alarm **6–12 hours BEFORE** sepsis becomes obvious, so doctors can treat earlier and save lives.

### 1.4 The Numbers

- **49 million** people get sepsis worldwide every year
- **11 million** die from it (more than cancer in many countries)
- It's the #1 cause of death in ICUs
- In Australia alone, sepsis kills ~5,000 people per year

---

## Part 2: The Data

### 2.1 Where Does Our Data Come From?

We use **MIMIC-IV** (Medical Information Mart for Intensive Care, version 4). It's a massive, free, publicly available dataset from **Beth Israel Deaconess Medical Center** in Boston, USA.

It contains real patient records (with names and identifying info removed for privacy) from their ICU. Thousands of patients, millions of measurements.

To access it, you need to:
1. Complete an ethics course (CITI training)
2. Sign a data use agreement
3. Get credentialed on PhysioNet (the platform that hosts it)

### 2.2 What's IN the Raw Data?

Imagine walking into an ICU and looking at everything recorded about a patient. That's what MIMIC-IV contains — but it's messy and spread across many tables:

**Vital Signs (chartevents):**
These are the numbers on the bedside monitor. Recorded every few minutes.
- **Heart Rate (HR):** How fast the heart is beating. Normal: 60–100 bpm. Sepsis often causes tachycardia (fast heart rate, 120+).
- **Respiratory Rate (RR):** Breaths per minute. Normal: 12–20. High RR means the body is struggling to get oxygen.
- **Temperature (Temp):** Normal ~37°C. Fever (>38°C) suggests infection. Very low temp (<36°C) can also indicate sepsis.
- **Blood Pressure (SBP/DBP/MAP):** The pressure of blood in your arteries. Low blood pressure in sepsis means organs aren't getting blood.
  - SBP = Systolic (when heart squeezes)
  - DBP = Diastolic (when heart relaxes)
  - MAP = Mean Arterial Pressure (the average)
- **Oxygen Saturation (SpO₂):** What percentage of your blood is carrying oxygen. Normal: 95–100%. Low = organs are oxygen-starved.

**Lab Tests (labevents):**
These require taking a blood sample and sending it to a lab. Results come back in 30min–2hrs. Doctors only order them when they suspect something — so there are **lots of gaps** (40–60% missing at any given hour).

Key labs for sepsis:
- **Lactate:** A waste product that builds up when organs don't get enough oxygen. Normal <2 mmol/L. High lactate (>4) is a sepsis red flag.
- **White Blood Cell Count (WBC):** Immune cells. Very high (infection fighting) or very low (immune exhaustion) suggests sepsis.
- **Creatinine:** Kidney waste product. High = kidneys are failing.
- **Bilirubin:** Liver waste product. High = liver is failing.
- **Platelets:** Blood clotting cells. Low = body is consuming them fighting inflammation.
- **Blood Gases (pH, PaO₂, PaCO₂):** How well the lungs are working and whether the blood is becoming acidic (bad).

...and more (BUN, sodium, potassium, glucose, etc.) — 17 lab features total.

### 2.3 Why Is the Raw Data Messy?

1. **Different codes for the same thing:** Heart Rate might be stored as `itemid 220045` in one table or `itemid 211` in another. You need a mapping to know which code means what.
2. **Different units:** Temperature might be in Fahrenheit or Celsius. FiO₂ might be 0.21 (fraction) or 21 (percentage).
3. **Irregular timing:** Vitals come every 1–5 minutes (way too frequent for our model). Labs come every 6–24 hours (very sparse).
4. **Missing values:** If a doctor didn't order a lactate test at hour 10, that value simply doesn't exist. It's not zero — it's unknown.

### 2.4 Data Harmonization — Cleaning the Mess

"Harmonization" is our process of turning this raw mess into a clean, usable table:

```
RAW MIMIC-IV (messy, millions of rows, different codes and units)
    ↓
Step 1: MAP item codes → standard names
    itemid 220045 → "hr"
    itemid 223761 → "temp"
    ↓
Step 2: CONVERT units
    98.6°F → 37°C
    FiO2 21% → 0.21
    ↓
Step 3: BIN into hourly buckets
    Vitals: take the median of all readings in that hour
    Labs: take the last (most recent) value in that hour
    ↓
Step 4: FORWARD-FILL missing values
    If no new temp reading at hour 5, carry forward hour 4's value
    (Vitals: up to 6 hours. Labs: up to 24 hours)
    ↓
Step 5: CALCULATE SOFA scores (see Part 3)
    ↓
Step 6: LABEL each time point (sepsis or not)
    ↓
CLEAN DATASET: one row per patient per hour, 24 features, sepsis label
```

**The output:** An HDF5 file (`mimic_processed_large.h5`) with 3,559 patients and 422,149 hourly observations.

### 2.5 What Does the Final Clean Data Look Like?

Each row:

| subject_id | charttime | hr | resp | temp | sbp | dbp | map | o2sat | lactate | wbc | creatinine | ... | sepsis_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10001 | hour 1 | 82 | 16 | 37.0 | 120 | 70 | 87 | 98 | 1.2 | 8.5 | 0.9 | ... | 0 |
| 10001 | hour 2 | 85 | 17 | 37.1 | 118 | 68 | 85 | 97 | NaN | NaN | NaN | ... | 0 |
| 10001 | hour 3 | 110 | 22 | 38.5 | 95 | 55 | 68 | 93 | 4.1 | 15.2 | 2.1 | ... | 1 |

Notice hour 2 has NaN for labs — no lab test was ordered that hour. And hour 3 looks scary: fast heart rate, high temp, low blood pressure, high lactate — classic sepsis.

---

## Part 3: How We Define Sepsis (Sepsis-3 Criteria)

### 3.1 The Problem of Labels

Our model needs to learn from examples: "this patient had sepsis, this one didn't." But there's no column in the hospital database that says "SEPSIS: YES/NO." We have to **compute it ourselves** from the clinical data.

### 3.2 The Sepsis-3 Definition

The international medical community agreed on this definition in 2016:

> **Sepsis = Suspected Infection + Organ Dysfunction**

**Part A — Suspected Infection:**
How do we know doctors suspected an infection? We look for TWO things happening within 24 hours of each other:
1. Doctor ordered **antibiotics** (they're treating a suspected infection)
2. Doctor ordered a **culture** — blood culture, urine culture, etc. (they're trying to identify what bug is causing it)

If both happened within 24 hours → doctors suspected infection.

**Part B — Organ Dysfunction (SOFA Score):**
SOFA = Sequential Organ Failure Assessment. It scores how badly each organ system is failing:

| Organ | What We Measure | Score 0 (healthy) | Score 4 (failing) |
|---|---|---|---|
| **Lungs** | PaO₂/FiO₂ ratio | ≥400 | <100 |
| **Blood clotting** | Platelet count | ≥150,000 | <20,000 |
| **Liver** | Bilirubin | <1.2 mg/dL | >12 mg/dL |
| **Heart/circulation** | Blood pressure | MAP ≥70 | Needs high-dose drugs |
| **Brain** | Glasgow Coma Scale | 15 (alert) | <6 (unresponsive) |
| **Kidneys** | Creatinine | <1.2 mg/dL | >5.0 mg/dL |

Total SOFA ranges from **0** (all organs healthy) to **24** (all organs failing).

**The rule:** If a patient's SOFA score **increases by 2 or more points** from their baseline → organ dysfunction is happening.

**Sepsis onset** = the moment when BOTH conditions are true (suspected infection AND SOFA increased by ≥2).

### 3.3 How We Use This for Labelling

```
Hour 1-10: Patient seems OK. SOFA baseline = 2. Label = 0 (no sepsis)
Hour 11: Doctor orders antibiotics + blood culture (suspected infection!)
Hour 15: SOFA jumps to 5 (delta = 3, which is ≥ 2) → SEPSIS ONSET
Hour 9-14: We label as "1" (positive) — this is the EARLY WARNING window
         (6-12 hours before things got really bad)
```

We label the window **before** onset because we want our model to **predict** sepsis before it's obvious. If we only labelled the moment of onset, the model would just detect "patient is already in sepsis" which isn't useful.

---

## Part 4: The Model — Our Multi-Agent System

### 4.1 Why Not Just Use a Simple Model?

You might ask: "Why not just throw all 24 features into a simple model and let it figure it out?"

The problem is that our data is **heterogeneous** (mixed types):

| Data Type | Completeness | Speed of Change | Best Approach |
|---|---|---|---|
| Vital signs | >95% complete | Changes every hour | Sequential model (LSTM) |
| Lab values | 40-60% MISSING | Changes every 6-24h | Needs special missing-value handling |
| Trends | Computed from above | Shows direction | Needs ability to compare distant time points |

One model trying to handle all three would need to be a jack-of-all-trades, master of none. Instead, we built **three specialist models** (agents), each designed for its data type.

### 4.2 What is a Neural Network? (The Absolute Basics)

If you've never seen a neural network before:

A neural network is a function that takes numbers in and produces numbers out. Inside, it has **weights** (thousands of adjustable knobs). During training, we show it examples with known answers, and it automatically adjusts its knobs to get better at producing correct answers.

```
Input numbers → [thousands of adjustable knobs] → Output number
                     (learned during training)
```

For our case:
- **Input:** 24 hours of patient data (vitals + labs)
- **Output:** A probability from 0 to 1 (0 = definitely no sepsis, 1 = definitely sepsis)
- **Knobs:** 312,419 parameters that get tuned during training

### 4.3 What is an LSTM?

**LSTM = Long Short-Term Memory.** It's a neural network designed for **sequences** — data where order matters.

Regular neural networks see all inputs at once, like looking at a photo. An LSTM reads inputs **one step at a time**, like reading a book — it remembers what came before.

For patient data:
```
Hour 1 → LSTM reads it, remembers "HR was 80, normal"
Hour 2 → LSTM reads it, remembers "HR still 80, boring"
...
Hour 15 → LSTM reads it, remembers "HR jumped to 140! After being stable!"
Hour 16 → LSTM reads it, knows "HR still 140, this isn't a one-off spike"
```

The LSTM can learn patterns like "a SUDDEN increase is more concerning than a value that's been high all along."

**Bi-directional LSTM** reads the sequence both forwards (hour 1→24) AND backwards (hour 24→1). This gives it context from both directions — it knows what comes before AND after each time point.

### 4.4 What is Attention?

After the LSTM reads all 24 hours, we have 24 hidden states (one per hour). Which hours matter most for predicting sepsis?

**Attention** is a mechanism that assigns a **weight** (importance score) to each hour. Hours with dramatic changes get high weights. Stable, boring hours get low weights.

```
Hour 1:  weight 0.02  (normal, boring)
Hour 2:  weight 0.02  (normal, boring)
...
Hour 15: weight 0.25  (HR spiked! Important!)
Hour 16: weight 0.20  (HR still high! Important!)
...
Hour 24: weight 0.03  (stable)
```

The final output is a weighted sum — the model pays 25% attention to hour 15, 20% to hour 16, and barely anything to the boring hours.

### 4.5 What is a Transformer?

A **Transformer** is an architecture (from the famous "Attention Is All You Need" paper, 2017 — the same tech behind ChatGPT).

Unlike LSTM which reads sequentially (hour 1, then 2, then 3...), a Transformer can compare **any two time points directly**. It's like having a bird's-eye view of the entire sequence at once.

This is great for spotting patterns like:
- "Lactate at hour 20 is much higher than hour 14" (6-hour comparison)
- "Blood pressure is dropping AND lactate is rising simultaneously" (cross-feature comparison)

LSTM would need to "remember" hour 14 all the way through hours 15-19 to compare it with hour 20. Transformers just look at both directly.

### 4.6 What is Imputation?

**Imputation** = filling in missing values.

Simple imputation methods:
- **Mean imputation:** Fill blanks with the average value across all patients. Problem: ignores context.
- **Forward fill:** Use the last known value. Problem: a lab from 12 hours ago might be outdated.
- **Zero fill:** Replace NaN with 0. Problem: 0 is a meaningful (and often dangerous) value for many labs.

**Learned imputation** (what we do): The model learns, during training, what the best fill-in value should be. It discovers things like:
- "When other kidney markers are bad, missing creatinine should be estimated as HIGH, not average"
- "When a patient is on vasopressors, missing lactate should be estimated as elevated"

The model also keeps a **mask** (a record of which values were real vs. guessed), so it knows to be less confident about its guesses.

### 4.7 Our Three Agents

Now putting it all together:

```
┌─────────────────────────────────────────────────────────────┐
│                    24-Hour Patient Window                     │
│  7 vital signs (every hour) + 17 lab values (sparse)         │
└────────────┬──────────────────┬──────────────────┬───────────┘
             │                  │                  │
             ▼                  ▼                  ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │  VITALS AGENT   │ │   LABS AGENT    │ │  TREND AGENT    │
   │                 │ │                 │ │                 │
   │ Bi-LSTM reads   │ │ Learns smart    │ │ Computes rates  │
   │ vitals forward  │ │ fill-in values  │ │ of change and   │
   │ & backward.     │ │ for missing     │ │ acceleration.   │
   │                 │ │ labs. Tracks    │ │                 │
   │ Attention       │ │ what's real     │ │ Transformer     │
   │ highlights      │ │ vs. guessed.    │ │ compares any    │
   │ critical hours. │ │                 │ │ two time points │
   │                 │ │ LSTM reads      │ │ directly.       │
   │ Input: 7 vitals │ │ the sequence.   │ │                 │
   │ Output: 64-dim  │ │                 │ │ Input: 24 trend │
   │ embedding       │ │ Input: 17 labs  │ │ features        │
   │                 │ │ + missing mask  │ │ Output: 64-dim  │
   │                 │ │ Output: 64-dim  │ │ embedding       │
   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
            │                   │                    │
            └─────────┬─────────┘────────────────────┘
                      ▼
           ┌─────────────────────┐
           │    META-LEARNER     │
           │                     │
           │ Receives 3 embeddings│
           │ (64-dim each).      │
           │                     │
           │ Learns how much to  │
           │ trust each agent    │
           │ FOR THIS PATIENT.   │
           │                     │
           │ Output:             │
           │ • Sepsis probability│
           │ • Agent weights     │
           │   (e.g., 35%/42%/23%)│
           └─────────────────────┘
```

### 4.8 What is an Embedding?

You'll see the word "embedding" a lot. It just means a **compressed summary** — a list of numbers that captures the essential information.

Each agent reads its data and produces a **64-dimensional embedding** — a list of 64 numbers that summarizes everything that agent learned from this patient. Think of it as a 64-number "report" from each specialist.

These numbers aren't human-readable (they're not "heart rate = 90, temperature = 38"). They're abstract learned features — the model decided internally what information to encode.

### 4.9 What is the Meta-Learner?

The meta-learner is the **decision maker**. It takes the three 64-number reports and:

1. **Computes attention weights** — decides how much to trust each agent for this specific patient
2. **Combines** the reports using those weights
3. **Outputs** a single sepsis probability (0 to 1)

The attention weights are **different for every patient**. This is the key insight:
- Patient with complete labs showing clear sepsis markers → Labs Agent gets 45% weight
- Patient with sparse labs but alarming vital signs → Vitals Agent gets 50% weight

### 4.10 What is Focal Loss?

During training, the model makes predictions and gets a "grade" on how wrong it was. This grade is called the **loss function** — lower loss = better predictions.

Normal loss (Binary Cross-Entropy) treats all mistakes equally. But in our dataset:
- 67% of samples are non-sepsis
- 33% are sepsis

If the model just guessed "no sepsis" every time, it'd be right 67% of the time! That's a bad model, but normal loss wouldn't penalize it enough.

**Focal Loss** fixes this by:
- Giving **more punishment** for mistakes on hard-to-classify examples
- Giving **less punishment** for easy, obvious cases
- The α parameter (0.25) balances the classes
- The γ parameter (2.0) controls how much to focus on hard examples

---

## Part 5: Training the Model

### 5.1 What Does "Training" Mean?

Training is the process of adjusting the model's 312,419 parameters to make better predictions:

```
Repeat for many rounds (epochs):
    1. Show the model a batch of 32 patients
    2. Model predicts sepsis probability for each
    3. Compare predictions to actual labels → compute loss (error)
    4. Calculate which direction to adjust each parameter to reduce error
    5. Adjust parameters by a small amount (learning rate)
```

Each full pass through the training data = one **epoch**. We train for up to 50 epochs.

### 5.2 Training vs. Validation vs. Test

We split our 3,559 patients into three groups:

| Set | Size | Role | Analogy |
|---|---|---|---|
| **Training** (70%) | 2,493 patients | Model learns from this | Textbook + homework |
| **Validation** (10%) | 356 patients | Monitor progress, decide when to stop | Practice exam |
| **Test** (20%) | 710 patients | Final evaluation, used ONCE | Final exam |

The model **never sees** validation or test data during learning. This prevents **overfitting** (memorizing answers instead of learning patterns).

**Early stopping:** We monitor performance on the validation set. If it stops improving for 10 epochs in a row (patience = 10), we stop training and keep the best version. This prevents the model from memorizing the training data.

### 5.3 What is Learning Rate?

Learning rate = how big of an adjustment we make to the parameters after each batch.

- **Too high:** Parameters jump around wildly, never settling on good values. Like trying to thread a needle while your hands are shaking violently.
- **Too low:** Parameters barely change, training takes forever. Like trying to walk to Sydney one millimeter at a time.
- **Just right:** Steady convergence to good values.

**Our key finding:** When we increased our data from 725 → 3,559 patients, we had to **reduce** the learning rate from 0.001 to 0.0001. More data means more stable gradients (directions), so you need smaller steps to avoid overshooting. This is discussed in detail in Q7 of the QnA document.

### 5.4 What is an Optimizer?

The optimizer is the algorithm that decides how to adjust the parameters. We use **AdamW** — it's like a smart version of basic gradient descent:

- It adjusts the learning rate individually for each parameter
- It uses **momentum** (remembers which direction it's been going, like a ball rolling downhill)
- The "W" means **weight decay** — it slightly shrinks all parameters each step, preventing any from getting too large (a form of regularization)

### 5.5 What is Dropout?

Dropout is a trick to prevent overfitting. During training, we randomly "turn off" 30% of the neurons in each layer. This forces the model to not rely on any single neuron — it must learn robust patterns that work even with some neurons missing.

During testing, all neurons are active (but their outputs are scaled down to compensate).

Think of it like a sports team practicing with random players sitting out each session. When the full team plays in the actual game, they're stronger because each player learned to contribute independently.

### 5.6 Our Training Experiments

We ran 6 experiments, each changing one thing:

```
v1: 725 patients, LR=0.001          → AUROC 0.7391  ✓ Good baseline
v2: 3559 patients, LR=0.001         → AUROC 0.6743  ✗ CRASHED (LR too high for more data)
v3: 3559 patients, LR=0.0001        → AUROC 0.7109  ✓ Fixed! (this is our best model)
v4: Changed focal loss alpha         → AUROC 0.6912  ✗ Worse
v5: Increased dropout to 0.4         → AUROC 0.7198  ~ Similar
v6: Smaller model (32 hidden, 1 layer)→ AUROC 0.7204  ~ Similar
```

**Takeaway:** The most impactful change was learning rate. Architecture changes (v4-v6) didn't help much — the model design was already good.

---

## Part 6: Evaluating the Model

### 6.1 Why Not Just Use "Accuracy"?

Accuracy = (correct predictions) / (total predictions).

If 67% of patients don't have sepsis, a model that always predicts "no sepsis" gets 67% accuracy. That's terrible — it would miss every single sepsis case — but accuracy makes it look okay.

We need metrics that aren't fooled by class imbalance.

### 6.2 AUROC — Our Primary Metric

**AUROC = Area Under the Receiver Operating Characteristic Curve**

Step by step:
1. Our model outputs a probability (e.g., 0.73) for each patient
2. We need a **threshold** to decide: above this → predict sepsis, below → no sepsis
3. At threshold 0.5: some patients are correctly flagged (true positives), some are missed (false negatives)
4. At threshold 0.3: we catch more sepsis cases but also raise more false alarms
5. The **ROC curve** plots this trade-off at EVERY possible threshold

The **area under** this curve = AUROC:
- **0.5** = the diagonal line = random guessing = useless
- **0.7** = decent (our model)
- **0.8** = good
- **0.9** = excellent
- **1.0** = perfect (never achieved in practice)

**Our AUROC: 0.7109** — this means if you randomly pick one sepsis patient and one non-sepsis patient, our model gives the sepsis patient a higher risk score **71% of the time**.

### 6.3 Sensitivity, Specificity, and the Trade-off

Once you pick a specific threshold, you get:

| Metric | Definition | Our Result | Meaning |
|---|---|---|---|
| **Sensitivity (Recall)** | Of all sepsis patients, how many did we catch? | 91.6% | We catch 92 out of 100 sepsis cases |
| **Specificity** | Of all non-sepsis patients, how many did we correctly ignore? | 33.7% | We incorrectly alarm on 66 out of 100 healthy patients |
| **PPV (Precision)** | Of all alarms we raised, how many were real? | 52.4% | About half our alarms are real sepsis |
| **NPV** | Of all patients we said were fine, how many were actually fine? | 83.3% | When we say "no sepsis," we're right 83% of the time |

### 6.4 Why We Chose High Sensitivity

In the ICU, there's an asymmetry:
- **Missing a sepsis case** → patient might die (catastrophic)
- **False alarm** → nurse checks the patient, sees they're fine, moves on (annoying but not harmful)

So we deliberately chose a low threshold (0.287) that catches almost all sepsis cases (91.6%), even though it means more false alarms. This is the **clinically appropriate** trade-off.

### 6.5 F1 Score

F1 balances precision and sensitivity into one number:

```
F1 = 2 × (Precision × Sensitivity) / (Precision + Sensitivity)
   = 2 × (0.524 × 0.916) / (0.524 + 0.916)
   = 0.666
```

F1 ranges from 0 (worst) to 1 (perfect). Our 0.666 means we have a reasonable balance — high sensitivity with acceptable precision.

### 6.6 AUPRC — Precision-Recall Curve

Similar to AUROC but focuses specifically on how well we find positive (sepsis) cases. A random model scores 0.331 (the class prevalence). Our model scores **0.6389** — nearly double random, meaning we're much better than chance at identifying sepsis.

### 6.7 Baseline Comparisons

We compared our model against simpler approaches to justify its complexity:

| Model | What it is | AUROC | Verdict |
|---|---|---|---|
| **Logistic Regression** | A straight line separating sepsis from non-sepsis | 0.6823 | Too simple for this problem |
| **Random Forest** | Many decision trees voting together | 0.6987 | Better, but can't handle temporal patterns |
| **XGBoost** | State-of-the-art tree-based model, wins most competitions | 0.7104 | Very strong — almost matches ours |
| **Simple MLP** | A basic neural network without LSTM/attention/agents | 0.6654 | Shows that architecture matters, not just "using deep learning" |
| **Multi-Agent (Ours)** | Three specialized agents + meta-learner | **0.7109** | Best, with added interpretability |

**Key point:** We only beat XGBoost by a tiny margin (0.0005 AUROC). But our model has a huge advantage: **interpretability**. We can explain WHY it made each prediction through agent attention weights. XGBoost is a black box.

---

## Part 7: Interpretability — Why Our Model Can Explain Itself

### 7.1 The Black Box Problem

Most deep learning models are "black boxes" — you put data in, a prediction comes out, and nobody knows why. Doctors won't trust a system that says "this patient has 78% sepsis risk" but can't explain why.

### 7.2 How Our Model Explains Itself

Our model provides two levels of explanation:

**Level 1: Agent Weights**
"I predicted sepsis because I relied 42% on lab values, 35% on vital signs, and 23% on trends."

This tells the doctor: "The labs were the biggest factor." The doctor can then look at the labs and see "oh, lactate is 4.2 — that makes sense."

**Level 2: Temporal Attention**
Within each agent, attention weights show WHICH hours mattered most: "The Vitals Agent focused most on hours 18–22, when heart rate was sustained above 130."

### 7.3 What Our Model Learned (Agent Weights)

Across all test patients:

| Agent | Sepsis Patients | Non-Sepsis Patients | What This Means |
|---|---|---|---|
| Labs | **41.3%** | 36.2% | For sepsis cases, the model relies more on labs — makes sense because lactate, WBC, creatinine are the gold-standard sepsis markers |
| Vitals | 35.8% | 32.9% | Vitals are consistently important but less discriminating alone |
| Trends | 22.9% | **30.9%** | For NON-sepsis, trends matter more — "nothing is getting worse" is evidence against sepsis |

This pattern matches clinical reasoning, which builds trust in the system.

---

## Part 8: The Technical Infrastructure

### 8.1 Project File Structure

```
Sepsis/
├── config/
│   └── data_config.yaml          # All settings (item mappings, units, thresholds)
├── data/
│   └── processed/
│       └── mimic_harmonized/
│           └── mimic_processed_large.h5  # The final clean dataset
├── src/
│   ├── data/
│   │   ├── harmonization.py      # Raw MIMIC → clean table
│   │   ├── sofa_calculator.py    # SOFA score computation
│   │   └── labeling.py           # Sepsis-3 label assignment
│   └── models/
│       └── multi_agent.py        # All 4 model components
├── models/
│   └── v3_large_lr1e4/           # Saved best model weights
├── notebooks/
│   ├── MIMIC_IV_Preprocessing.ipynb         # Data processing pipeline
│   ├── MIMIC_IV_Preprocessing_Batched.ipynb # Batched version (larger data)
│   ├── Train_MultiAgent_Model.ipynb         # Model training + evaluation
│   ├── Complete_Metrics_Analysis.ipynb      # Detailed metrics + agent analysis
│   ├── Baseline_Comparison.ipynb            # Comparison with simpler models
│   └── Data_Exploration_Quick.ipynb         # Data exploration
└── docs/
    ├── PROJECT_REPORT_DRAFT.md    # Academic report
    ├── QnA.md                     # Supervisor Q&A prep
    └── PROJECT_WALKTHROUGH.md     # This file
```

### 8.2 Key Technologies Used

| Technology | What it is | Why we use it |
|---|---|---|
| **Python** | Programming language | Industry standard for ML/data science |
| **PyTorch** | Deep learning framework | Flexible, research-friendly, GPU-accelerated |
| **pandas** | Data manipulation library | For processing tabular patient data |
| **scikit-learn** | ML library | Baseline models, train/test splitting, metrics |
| **HDF5 (h5py)** | File format | Efficient storage for large numerical datasets |
| **XGBoost** | Gradient boosting library | Our strongest baseline model |
| **Jupyter Notebooks** | Interactive coding environment | For experiments, visualization, and analysis |

### 8.3 Hardware

Training was done on an **NVIDIA Tesla T4 GPU** (via cloud). Training the best model (v3) took approximately 2.5 hours for 50 epochs.

---

## Part 9: Summary — The Story in 60 Seconds

1. **Sepsis kills 11 million people/year.** Every hour of delay increases death risk by 7.6%. Early detection saves lives.

2. **We used MIMIC-IV data** — real ICU records from 3,559 patients, processed into clean hourly measurements (7 vital signs + 17 lab values).

3. **We built a multi-agent system** with three specialists:
   - Vitals Agent (bi-LSTM + attention) for continuous vital signs
   - Labs Agent (LSTM + learned imputation) for sparse lab values
   - Trend Agent (Transformer) for rates of change
   - Meta-Learner combines them with patient-specific weights

4. **Our key finding:** Learning rate must be reduced when scaling data (0.001 → 0.0001 when going from 725 → 3,559 patients).

5. **Results:** AUROC 0.7109, catching 91.6% of sepsis cases. Matches XGBoost but with the added benefit of **interpretability** — the model can explain which data sources drove each prediction.

6. **Next:** Train on the full MIMIC-IV dataset and validate on external hospital data.

---

*Last updated: March 2026*
