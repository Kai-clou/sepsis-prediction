# Q&A — Sepsis Prediction Project

Prepared for supervisor meeting / viva questions.

---

## Q1: What did we process our raw MIMIC data into? What's data harmonization? Does the SOFA calculator also play a role? Where are these files?

### What we started with (Raw MIMIC-IV)

Raw MIMIC-IV is messy hospital data spread across many tables:

- **chartevents** — vital signs recorded every 1–5 minutes, identified by cryptic `itemid` codes (e.g., itemid 220045 = Heart Rate)
- **labevents** — lab tests taken sporadically (every 6–24 hours), also coded by itemid
- **prescriptions** — medication orders (we need antibiotics for sepsis labelling)
- **microbiologyevents** — culture orders (blood cultures, urine cultures)
- **inputevents_mv** — IV drips like vasopressors (needed for cardiovascular SOFA)
- **outputevents** — urine output (needed for renal SOFA)

### What we turned it into (Harmonized Dataset)

We transformed all of the above into a single clean table stored as an HDF5 file (`.h5`), where:

- Each **row** = one patient, one hour
- **Columns**: `subject_id`, `hadm_id`, `charttime`, then 7 vitals + 17 labs + sepsis label
- **24-hour sliding windows** are used for training (each sample = 24 consecutive hours)

### What is Data Harmonization?

"Harmonization" means mapping the raw MIMIC-IV codes into a **standard set of variable names** that our model understands. For example:

| Raw MIMIC-IV | Harmonized Name |
|---|---|
| itemid 220045 | `hr` (Heart Rate) |
| itemid 220210 | `resp` (Respiratory Rate) |
| itemid 223761 (°F) | `temp` (Temperature, converted to °C) |
| itemid 50813 | `lactate` |

It also involves:
- **Unit conversion** — e.g., Temperature from Fahrenheit → Celsius, FiO2 from percentage → fraction
- **Temporal alignment** — binning irregular measurements into 1-hour buckets (vitals use median, labs use last value in the hour)
- **Forward-filling** — if no new measurement, carry the last known value forward (vitals up to 6h, labs up to 24h)

### Does the SOFA Calculator play a role?

**Yes — it's essential for creating our labels.** We don't have a column in MIMIC that says "this patient has sepsis." We have to compute it ourselves using the Sepsis-3 definition (see Q3). The SOFA calculator:

1. Scores 6 organ systems (0–4 each, total 0–24)
2. Calculates a **baseline SOFA** (minimum in first 24 hours)
3. Tracks **delta SOFA** (change from baseline) at every hour
4. Flags **organ dysfunction** when delta SOFA ≥ 2

This feeds into the labelling pipeline: **Sepsis = suspected infection + organ dysfunction (SOFA Δ ≥ 2)**.

### File Locations

| What | Where |
|---|---|
| Harmonization code | `src/data/harmonization.py` — class `MIMICHarmonizer` |
| SOFA calculator | `src/data/sofa_calculator.py` — class `SOFACalculator` |
| Sepsis labelling | `src/data/labeling.py` — class `SepsisLabeler` |
| Configuration (itemid mappings, units) | `config/data_config.yaml` |
| Preprocessing notebook | `notebooks/MIMIC_IV_Preprocessing.ipynb` (small batch) |
| Batched preprocessing | `notebooks/MIMIC_IV_Preprocessing_Batched.ipynb` (large batch) |
| Final processed dataset | `data/processed/mimic_harmonized/mimic_processed_large.h5` |
| Data dictionary | `docs/data_dictionary.md` |

---

## Q2: Explain like I'm 5 — what are the agents, the meta-learner, and where can I see them?

### Vitals Agent (Bi-directional LSTM with Attention)

**ELI5:** Imagine you're reading a patient's heart rate chart. Normally you read left-to-right (morning → night). But what if you could also read it right-to-left (night → morning)? That's **bi-directional** — you understand the story from both directions.

Now imagine you have a yellow highlighter. You go through the chart and highlight the **scary moments** — like when the heart rate suddenly jumped from 80 to 140. You don't highlight the boring stable parts. That's **attention** — the model learns which time points matter most and pays more attention to them.

- **Input:** 7 vital signs (HR, respiratory rate, temperature, SBP, DBP, MAP, SpO₂)
- **Why LSTM?** Vital signs are recorded nearly every hour (>95% complete), so a sequence model works well
- **Output:** A 64-number summary (embedding) of what the vitals are telling us

### Labs Agent (LSTM with Learned Imputation)

**ELI5:** Lab tests are like homework that the doctor only assigns sometimes. So you have a lot of **blank spaces** (40–60% missing). Simple approaches just fill blanks with the class average. But our model is smarter:

Imagine a student who's been getting bad grades in kidney tests. If their lactate homework is missing, our model guesses "probably bad" instead of "average." It **learns** the best guess for each blank based on the other answers it can see. It also remembers **which answers were real and which were guesses**, so it knows how confident to be.

- **Input:** 17 lab values (lactate, WBC, creatinine, platelets, etc.) + a missing-value mask
- **Why learned imputation?** Because 40–60% of lab values are missing and simple fill-in methods lose important context
- **Output:** A 64-number summary (embedding) of what the labs are telling us

### Trend Agent (Transformer Encoder)

**ELI5:** This agent doesn't care what the actual numbers are — it cares about **which way things are going**. A lactate of 4.0 that's been stable is very different from a lactate of 4.0 that was 2.0 six hours ago (that's scary — it's rising fast!).

It computes two things:
- **Rate of change** — is it going up or down?
- **Acceleration** — is the change speeding up or slowing down?

The Transformer architecture lets it compare **any two time points** directly (unlike LSTM which reads one step at a time). So it can spot patterns like "lactate rising AND blood pressure falling at the same time" — which is a classic sepsis red flag.

- **Input:** 24 derived features (rates of change + acceleration for all vitals and labs)
- **Why Transformer?** Can directly compare distant time points and spot multi-variable patterns
- **Output:** A 64-number summary (embedding) of the trends

### Meta-Learner (Attention-Weighted Fusion)

**ELI5:** The meta-learner is the **team leader**. It gets a report from each of the three agents and decides how much to trust each one.

If a patient has great lab data showing clear sepsis signs, the leader says "I'll listen 45% to Labs, 35% to Vitals, 20% to Trends." But for a different patient with sparse labs, it might say "Labs data is unreliable — I'll listen 50% to Vitals instead."

This dynamic weighting is what makes the system **interpretable** — you can see *which agent drove each prediction*.

- **Input:** Three 64-number embeddings (one from each agent)
- **Process:** Learns attention weights for each agent, per patient
- **Output:** A single sepsis probability (0 to 1) plus the agent weights

### Where to see these models in action

| What | Where |
|---|---|
| Model source code | `src/models/multi_agent.py` |
| Training + evaluation | `notebooks/Train_MultiAgent_Model.ipynb` (cells 20–32) |
| Meta-learner weights visible | `notebooks/Complete_Metrics_Analysis.ipynb` — agent weight analysis section |
| Baseline comparisons | `notebooks/Baseline_Comparison.ipynb` |

In the training notebook, after the model runs, the output dict contains:
```python
outputs = model(vitals, labs, labs_mask, all_features)
outputs['probability']     # sepsis risk (0-1)
outputs['agent_weights']   # [vitals_weight, labs_weight, trend_weight] per patient
outputs['logits']          # raw output before sigmoid
```

---

## Q3: What is the Sepsis-3 criteria?

Sepsis-3 is the **current international consensus definition** (Singer et al., 2016, published in JAMA). It replaced the older SIRS-based definitions.

### Definition

> **Sepsis = Suspected Infection + Organ Dysfunction**

Specifically:

1. **Suspected Infection** — The patient received **antibiotics** AND had a **culture ordered** (blood, urine, etc.) within a 24-hour window of each other. This is the clinical proxy for "the doctors thought there was an infection."

2. **Organ Dysfunction** — A **SOFA score increase of ≥ 2 points** from the patient's baseline. SOFA (Sequential Organ Failure Assessment) scores 6 organ systems:

| Organ System | What it measures | Score range |
|---|---|---|
| **Respiratory** | PaO₂/FiO₂ ratio | 0–4 |
| **Coagulation** | Platelet count | 0–4 |
| **Liver** | Bilirubin level | 0–4 |
| **Cardiovascular** | Mean arterial pressure + vasopressor use | 0–4 |
| **Neurological** | Glasgow Coma Scale | 0–4 |
| **Renal** | Creatinine + urine output | 0–4 |

**Total SOFA: 0–24** (higher = more organ failure)

### How we use it in our project

1. Calculate **baseline SOFA** = minimum SOFA in first 24 hours of ICU stay
2. At every hour, calculate **current SOFA**
3. Compute **delta SOFA** = current − baseline
4. If delta SOFA ≥ 2 AND suspected infection → **sepsis onset**
5. We label the time window **6–12 hours before onset** as positive (this is the early prediction window)

### Why Sepsis-3 and not older definitions?

- **SIRS criteria** (the old method) were too sensitive — things like exercise or anxiety could trigger them
- **Sepsis-3** focuses on organ dysfunction, which is more specific and clinically meaningful
- It's the standard used in modern sepsis research (including the PhysioNet 2019 Challenge)

---

## Q4: What is patient-level stratified splitting? Why the 70/10/20 split?

### Patient-Level Splitting

**The problem it solves:** One patient generates many 24-hour windows (samples). Patient A might have 50 windows, Patient B might have 30. If we split randomly by window, Patient A's windows could end up in both the training AND test set. The model would then "memorize" Patient A during training and score perfectly on Patient A's test windows — but this is **cheating** (data leakage).

**Patient-level splitting** means: all windows from the same patient go into the **same** split. If Patient A is in training, ALL of Patient A's data is in training. None leaks into test.

```
Patient A (50 windows) → ALL in Training
Patient B (30 windows) → ALL in Test
Patient C (40 windows) → ALL in Validation
```

### Stratified Splitting

**The problem it solves:** About 33% of our patients have sepsis. If we split randomly, we might accidentally get 50% sepsis in training and 10% in test — the model would train on a different distribution than it's tested on.

**Stratified** means we ensure the sepsis ratio is approximately the same in every split:

| Split | Patients | Sepsis Rate |
|---|---|---|
| Training (70%) | 2,493 | 32.4% |
| Validation (10%) | 356 | 32.8% |
| Test (20%) | 710 | 33.1% |

The rates are almost identical — that's stratification working.

### Why 70/10/20?

| Split | Purpose | Why this size? |
|---|---|---|
| **70% Training** | Model learns from this data | Needs the most data to learn patterns. 70% is standard and gives us 2,493 patients — enough to learn from |
| **10% Validation** | Used during training to decide when to stop (early stopping) and tune hyperparameters | Doesn't need to be large — just enough to get a reliable signal. 356 patients is sufficient |
| **20% Test** | Final evaluation — model NEVER sees this during training | Needs to be large enough for reliable metrics. 710 patients gives us statistically meaningful results |

**Why not 80/10/10?** A larger test set (20%) gives more confidence in our reported metrics. With 710 test patients instead of 355, our AUROC estimate is more reliable.

**Why not 60/20/20?** We'd lose training data. With only 3,559 patients total (medium batch), every training patient matters.

In code, this is done with:
```python
from sklearn.model_selection import train_test_split

# Split by PATIENT, stratify by PATIENT-LEVEL sepsis label
train_ids, test_ids = train_test_split(
    patient_ids, test_size=0.2, stratify=patient_labels, random_state=42
)
train_ids, val_ids = train_test_split(
    train_ids, test_size=0.125, stratify=train_labels, random_state=42
)
# 0.125 of 80% = 10% of total → gives us 70/10/20
```

---

## Q5: Why these baselines? What is AUROC? What is F1? How do they affect our experiment?

### Why we chose those specific baselines

| Baseline | Why included |
|---|---|
| **Logistic Regression** | The simplest possible classifier. If our complex model can't beat this, something is wrong. It's the "sanity check" |
| **Random Forest** | A strong traditional ML model that handles non-linear relationships. Widely used in clinical prediction studies |
| **XGBoost** | The "gold standard" of tabular ML — wins most Kaggle competitions. If we beat XGBoost, it's a strong result |
| **Simple MLP** | A basic neural network (no LSTM, no attention, no agents). Shows that the multi-agent architecture adds value beyond just "using deep learning" |

The progression tells a story: **simple linear → ensemble trees → gradient boosting → basic neural net → our multi-agent system**. Each step up adds complexity, and we need to justify that complexity by showing improvement.

### What is AUROC?

**AUROC = Area Under the Receiver Operating Characteristic curve** (value from 0 to 1)

**ELI5:** Imagine picking one random sepsis patient and one random non-sepsis patient. AUROC is the probability that our model gives the sepsis patient a **higher risk score** than the non-sepsis patient.

- **AUROC = 0.5** → coin flip, useless model
- **AUROC = 0.7** → 70% chance it ranks correctly
- **AUROC = 1.0** → perfect, always ranks correctly

**Why it's our main metric:**

1. **Threshold-independent** — We don't have to pick a cutoff point. AUROC evaluates the model across ALL possible thresholds. This matters because different hospitals might want different sensitivity/specificity trade-offs
2. **Handles class imbalance** — With 33% sepsis cases, accuracy would be misleading (predicting "no sepsis" for everyone gives 67% accuracy). AUROC isn't fooled by this
3. **Standard in clinical ML** — The PhysioNet 2019 Sepsis Challenge and most clinical papers report AUROC, so we can directly compare our results to published work
4. **Interpretable** — "Our model correctly ranks a sepsis patient above a non-sepsis patient 71% of the time" is easy to understand

### What is AUPRC?

**AUPRC = Area Under the Precision-Recall Curve** — our secondary metric.

It focuses on how well the model finds the positive (sepsis) cases. A random model would score 0.331 (the class prevalence). Our model scores 0.6389 — nearly double, meaning it's much better than random at finding sepsis cases.

### What is F1 Score?

**F1 = the harmonic mean of Precision and Recall** (value from 0 to 1)

- **Precision** = Of all the patients we flagged as sepsis, how many actually had sepsis? (Are our alarms real?)
- **Recall (Sensitivity)** = Of all actual sepsis patients, how many did we catch? (Are we missing cases?)
- **F1** = Balances both. It penalizes models that sacrifice one for the other.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Our best results (v6 at optimal F1 threshold):**
- Sensitivity: 84.3% → we catch 84% of sepsis cases
- Specificity: 52.7% → we correctly clear 53% of healthy patients
- **F1: 0.6999** → strong balanced score

### How does this affect our experiment?

The F1 score reveals a **clinical trade-off**. Across our 6 versions, we saw this evolve:

| Version | Sensitivity | Specificity | What's happening |
|---|---|---|---|
| v1 | 0.908 | 0.219 | "Everyone has sepsis!" — too many false alarms |
| v6 | 0.843 | 0.527 | Balanced — catches most sepsis, clears half of healthy |

The model traded a small drop in sensitivity (91% → 84%) for a huge gain in specificity (22% → 53%). In critical care, this is a better balance — **missing a sepsis case is dangerous**, but constant false alarms cause "alert fatigue" where clinicians start ignoring warnings.

If a hospital wanted even higher sensitivity, they could lower the threshold (more false alarms). If they wanted fewer false alarms, raise the threshold (might miss some cases). This is a deployment decision, not a model problem.

---

## Q6: What are agent attention weights?

### The concept

Agent attention weights are the **importance scores** the meta-learner assigns to each of the three agents for every individual prediction. They always sum to 1 (100%).

Think of it like a panel of three expert doctors voting on a diagnosis:
- Dr. Vitals says "I'm 70% sure it's sepsis"
- Dr. Labs says "I'm 85% sure it's sepsis"
- Dr. Trends says "I'm 40% sure it's sepsis"

The meta-learner doesn't just average their opinions. It decides **how much to trust each doctor** for this specific patient:
- "Dr. Labs has strong evidence (complete labs, high lactate) → I'll weight her opinion at 45%"
- "Dr. Vitals has useful info (tachycardia) → 35%"
- "Dr. Trends doesn't have much to say → 20%"

Final prediction = 0.45 × Labs_opinion + 0.35 × Vitals_opinion + 0.20 × Trends_opinion

### What our model learned

| Agent | Overall Weight | For Sepsis Cases | For Non-Sepsis Cases |
|---|---|---|---|
| Vitals Agent | 34.2% | 35.8% | 32.9% |
| **Labs Agent** | **38.5%** | **41.3%** | 36.2% |
| Trend Agent | 27.3% | 22.9% | 30.9% |

**Key insights:**
- **Labs Agent dominates** (38.5% overall) — makes clinical sense because lab values like lactate, WBC, and creatinine are the strongest sepsis indicators
- **Labs weight increases for sepsis cases** (41.3% vs 36.2%) — when the model sees sepsis, it relies even more on lab evidence
- **Trend Agent is more useful for ruling OUT sepsis** (30.9% for non-sepsis vs 22.9% for sepsis) — stable trends = "nothing is getting worse" = less likely sepsis

### Why this matters

This is the **interpretability advantage** of our multi-agent architecture. Unlike a black-box model that just outputs a number, our system can explain: "I predicted sepsis because the Labs Agent (41% weight) detected elevated lactate at hour 20, and the Vitals Agent (36% weight) saw sustained tachycardia during hours 18–22."

This helps clinicians **trust** the system because the explanation matches how they think about sepsis diagnosis.

### Where to see this in code

```python
# In src/models/multi_agent.py — MetaLearner class
outputs = model(vitals, labs, labs_mask, all_features)
weights = outputs['agent_weights']  # shape: (batch_size, 3)
# weights[:, 0] = vitals weight
# weights[:, 1] = labs weight
# weights[:, 2] = trend weight
```

Visualized in `notebooks/Complete_Metrics_Analysis.ipynb` in the agent weight analysis section.

---

## Q7: What is learning rate? How did it affect our experiment?

### What is learning rate?

**ELI5:** Imagine you're blindfolded on a hilly landscape, trying to find the lowest valley (the best model). At each step, you feel which direction slopes downward (that's the gradient) and take a step. The **learning rate** is **how big your steps are**.

- **Too large (1×10⁻³ for big data):** You take huge steps and keep overshooting the valley — you jump right over it and land on the other side. The model never settles on good parameters.
- **Too small:** Tiny baby steps — you'll eventually get there but it takes forever, and you might get stuck in a shallow dip instead of finding the deepest valley.
- **Just right (1×10⁻⁴ for our data):** You take measured steps, steadily approaching the lowest point.

### How it affected our experiment — the key story

This was our **most important experimental finding**:

| Version | Patients | Learning Rate | AUROC | AUPRC | F1 | Sens | Spec | What happened |
|---|---|---|---|---|---|---|---|---|
| **v1** | 725 | 1×10⁻³ | **0.6478** | 0.4984 | 0.5625 | 0.908 | 0.219 | LR too high even for small data |
| **v2** | 3,559 (5× more) | 1×10⁻³ | **0.7136** | 0.6627 | 0.6758 | 0.865 | 0.418 | More data helps, but LR still too high |
| **v3** | 3,559 | 1×10⁻⁴ (10× smaller) | **0.7361** | 0.6657 | 0.6943 | 0.853 | 0.495 | LR fix → big improvement |

### Why did lowering the LR help?

When we increased from 725 → 3,559 patients, the **gradients became more stable** (more data = more reliable direction estimate). But the learning rate was still calibrated for noisy small-data gradients. With stable gradients and a large step size, the optimizer kept **overshooting** the optimal parameters — like taking a running leap when you should be carefully stepping.

Reducing the learning rate by 10× (from 1×10⁻³ to 1×10⁻⁴) gave the optimizer smaller, more precise steps. With the stable gradients from 3,559 patients AND small steps, it could carefully converge to a good solution.

### The practical takeaway

> **When you scale up your dataset, scale down your learning rate.**

This is a well-known principle in deep learning, but our experiment demonstrated it concretely:
- The relationship isn't always linear (we went 5× data, 10× smaller LR)
- Default hyperparameters from small experiments often break on larger data
- Always re-tune hyperparameters when scaling — don't assume what worked on a small dataset transfers directly

### What we tried after v3

| Version | Change from v3 | AUROC | AUPRC | F1 | Sens | Spec | Conclusion |
|---|---|---|---|---|---|---|---|
| v4 | Focal alpha 0.25 → 0.35 | 0.7372 | 0.6698 | 0.6928 | 0.844 | 0.504 | Marginal gain, slightly better specificity |
| v5 | Dropout 0.3 → 0.4 | 0.7348 | 0.6676 | 0.6955 | 0.847 | 0.508 | Almost identical — model wasn't overfitting |
| **v6** | Hidden 64→32, Layers 2→1 | **0.7382** | 0.6530 | **0.6999** | 0.843 | **0.527** | **Best! Simpler model generalizes better** |

**Key insight:** v6 (the smallest model) is the best. With only 3,559 patients, a smaller model (32 hidden units, 1 layer) avoids overfitting better than the larger architecture (64 hidden, 2 layers). It also trains 17% faster (34m vs 41m).

---

## Q8: What is Dropout? What is Focal Alpha? How did they affect our experiment?

### Dropout — preventing the model from memorizing

**ELI5:** Imagine a group project where one student does all the work. If that student is sick on exam day, the whole team fails. Dropout prevents this by **randomly kicking out team members during practice**, forcing everyone to learn the material.

In neural networks, during each training step, dropout randomly **turns off** a percentage of neurons:

```
dropout = 0.3 → 30% of neurons randomly set to zero each step
dropout = 0.4 → 40% of neurons randomly set to zero each step

Step 1:  [neuron1] [  OFF  ] [neuron3] [  OFF  ] [neuron5]
Step 2:  [  OFF  ] [neuron2] [neuron3] [neuron4] [  OFF  ]
Step 3:  [neuron1] [neuron2] [  OFF  ] [neuron4] [neuron5]
```

This forces the network to **not rely on any single neuron** — every neuron must contribute. The result is a more robust model that generalizes better to new patients.

At test time, dropout is turned off (all neurons active), but their outputs are scaled down to compensate.

**In our experiment (v3 vs v5):**
- v3 (dropout=0.3): AUROC 0.7361
- v5 (dropout=0.4): AUROC 0.7348

Almost no difference! This tells us the model **wasn't overfitting much** with dropout=0.3 already, so increasing it didn't help. The training data (3,559 patients) was enough for the model to learn real patterns rather than memorize noise.

### Focal Alpha — paying more attention to the minority class

**ELI5:** Imagine a teacher grading 100 essays — 67 are "pass" and 33 are "fail." If the teacher just marks everything as "pass," they get 67% accuracy! But they miss every failing student. **Focal alpha** tells the teacher: "Pay extra attention to the failing papers."

In our FocalLoss function, alpha controls how much the model **cares about each class**:

```
alpha = 0.25 → 25% weight on sepsis cases, 75% weight on non-sepsis
alpha = 0.35 → 35% weight on sepsis cases, 65% weight on non-sepsis
```

Higher alpha = the model gets penalized more for missing sepsis cases.

**In our experiment (v3 vs v4):**
- v3 (alpha=0.25): AUROC 0.7361, Specificity 0.495
- v4 (alpha=0.35): AUROC 0.7372, Specificity 0.504

Tiny gain (+0.001 AUROC). Why so small? Because our dataset isn't extremely imbalanced — it's 33% sepsis / 67% non-sepsis. Focal alpha matters more when you have something like 5% positive / 95% negative. At 33/67, the default alpha already handles it well.

### Note: Focal Loss also has Gamma (γ = 2.0)

We kept gamma fixed at 2.0 across all experiments. Gamma controls how much the loss **ignores easy examples**:

```
gamma = 0 → standard cross-entropy (treats all samples equally)
gamma = 2 → downweights easy examples, focuses on hard-to-classify cases
```

A patient who is obviously septic (probability = 0.99) contributes almost nothing to the loss — the model already knows. A borderline case (probability = 0.55) contributes much more. This helps the model focus its learning on the tricky cases.

---

*Last updated: March 2026*
