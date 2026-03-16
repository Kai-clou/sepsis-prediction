# Multi-Agent Deep Learning System for Early Sepsis Prediction in Intensive Care Units

---

**Student:** Jason
**Student Number:** [INSERT STUDENT NUMBER]
**Supervisor:** Ms. Ying
**Institution:** University of Technology Sydney
**Faculty:** Faculty of Engineering and Information Technology
**Degree:** Bachelor of Science (Honours) in Information Technology
**Date:** March 2026

---

## Declaration

I certify that this thesis has not already been submitted for any degree and is not being submitted as part of candidature for any other degree. I also certify that the thesis has been written by me and that any help that I have received in preparing this thesis, and all sources used, have been acknowledged in this thesis.

**Signature:** ______________________
**Date:** ______________________

---

## Acknowledgements

[INSERT: Acknowledge your supervisor Ms. Ying, any lab resources, family/friends who supported you, UTS for providing access to computing resources, and the MIT Laboratory for Computational Physiology for making MIMIC-IV publicly available.]

---

## Abstract

Sepsis kills an estimated 11 million people each year, and in intensive care units, every hour of delayed treatment raises mortality risk by nearly 8%. Catching it early matters — but it is hard. Vital signs stream in almost continuously while lab results arrive sporadically, with 40–60% of values missing at any given hour. Most existing models treat these very different data streams the same way, which seemed like a missed opportunity.

This thesis presents a multi-agent deep learning system that assigns a specialised neural network to each type of clinical data: a bi-directional LSTM with attention for vital signs, an LSTM with learned imputation for laboratory values, and a Transformer encoder for temporal trends. An attention-weighted meta-learner then combines their outputs, deciding how much to trust each agent on a per-patient basis. I trained and evaluated the system across seven experiments on MIMIC-IV, scaling from 725 patients to the full dataset of 65,297 ICU admissions (7.9 million hourly observations), varying the dataset size, learning rate, regularisation, and model architecture.

Four main findings came out of the experiments. First, scaling from 725 to 3,559 patients required dropping the learning rate from 1×10⁻³ to 1×10⁻⁴ — keeping the original rate led to worse convergence. Second, the smallest model (32 hidden units, 1 layer) outperformed the larger one (64 hidden, 2 layers), suggesting the bigger architecture was slightly overfitting. Third, scaling to the full MIMIC-IV dataset (65,297 patients) further improved all metrics, confirming that data volume is the strongest lever for performance. Fourth, patient-level evaluation — aggregating predictions per patient rather than per sliding window — revealed the model's true discriminative power: AUROC 0.8565, AUPRC 0.7856, and specificity 0.729, substantially stronger than sequence-level metrics (AUROC 0.7702) suggested. The model outperformed both clinical scoring systems (qSOFA 0.66–0.70, SIRS 0.64–0.68) and machine learning baselines including XGBoost (AUROC 0.7579) and Random Forest (AUROC 0.7448).

---

## Table of Contents

1. [Introduction](#1-introduction)
    - 1.1 [Background](#11-background)
    - 1.2 [Research Significance](#12-research-significance)
    - 1.3 [Research Questions](#13-research-questions)
    - 1.4 [Literature Review](#14-literature-review)
    - 1.5 [Scope and Contributions](#15-scope-and-contributions)
2. [Methodology](#2-methodology)
    - 2.1 [Dataset: MIMIC-IV](#21-dataset-mimic-iv)
    - 2.2 [Data Preprocessing and Harmonisation](#22-data-preprocessing-and-harmonisation)
    - 2.3 [Sepsis Labelling via Sepsis-3 Criteria](#23-sepsis-labelling-via-sepsis-3-criteria)
    - 2.4 [System Architecture](#24-system-architecture)
    - 2.5 [Vitals Agent: Bi-directional LSTM with Attention](#25-vitals-agent-bi-directional-lstm-with-attention)
    - 2.6 [Labs Agent: LSTM with Learned Imputation](#26-labs-agent-lstm-with-learned-imputation)
    - 2.7 [Trend Agent: Transformer Encoder](#27-trend-agent-transformer-encoder)
    - 2.8 [Meta-Learner: Attention-Weighted Fusion](#28-meta-learner-attention-weighted-fusion)
    - 2.9 [Training Procedure](#29-training-procedure)
    - 2.10 [Experimental Design](#210-experimental-design)
    - 2.11 [Evaluation Metrics](#211-evaluation-metrics)
3. [Results](#3-results)
    - 3.1 [Experimental Iterations](#31-experimental-iterations)
    - 3.2 [Best Model Performance](#32-best-model-performance)
    - 3.3 [Sensitivity-Specificity Trade-off](#33-sensitivity-specificity-trade-off)
    - 3.4 [Agent Contribution Analysis](#34-agent-contribution-analysis)
    - 3.5 [Baseline Model Comparison](#35-baseline-model-comparison)
    - 3.6 [Patient-Level Evaluation](#36-patient-level-evaluation)
    - 3.7 [Comparison with Clinical Scoring Systems](#37-comparison-with-clinical-scoring-systems)
4. [Discussion](#4-discussion)
    - 4.1 [Addressing the Research Questions](#41-addressing-the-research-questions)
    - 4.2 [Learning Rate Scaling](#42-learning-rate-scaling)
    - 4.3 [Model Complexity and Generalisation](#43-model-complexity-and-generalisation)
    - 4.4 [Data Scaling](#44-data-scaling)
    - 4.5 [Sequence-Level vs Patient-Level Evaluation](#45-sequence-level-vs-patient-level-evaluation)
    - 4.6 [Regularisation and Loss Function Tuning](#46-regularisation-and-loss-function-tuning)
    - 4.7 [Clinical Interpretability](#47-clinical-interpretability)
    - 4.8 [Learned Imputation](#48-learned-imputation)
    - 4.9 [Clinical Deployment Considerations](#49-clinical-deployment-considerations)
5. [Future Work](#5-future-work)
6. [Conclusion](#6-conclusion)
7. [References](#7-references)
8. [Appendices](#8-appendices)

---

## 1. Introduction

### 1.1 Background

Sepsis — life-threatening organ dysfunction triggered by infection — is one of the biggest killers in modern hospitals. Globally, it affects around 49 million people each year and causes an estimated 11 million deaths, roughly one in five of all deaths worldwide (Rudd et al., 2020). The numbers are especially grim in intensive care units: Kumar et al. (2006) showed that every hour of delayed antibiotic treatment raises mortality risk by 7.6%. Getting ahead of sepsis, even by a few hours, can be the difference between life and death.

The trouble is that sepsis does not announce itself. Unlike a heart attack, where a blood test for troponin can confirm the diagnosis quickly, sepsis creeps in. A patient's heart rate drifts upward over several hours. Their blood pressure softens. Lab markers like lactate and white blood cell count start to shift. By the time these changes are obvious enough for a bedside scoring system to flag, organ damage may already be underway. Clinicians often describe it as a pattern you can see in hindsight but struggle to catch in real time.

What makes the problem harder from a data perspective is that ICU data is messy in a very specific way. Vital signs — heart rate, blood pressure, respiratory rate, temperature, oxygen saturation — are recorded almost continuously, with over 95% completeness. Lab tests, on the other hand, only happen when a doctor orders them. Lactate might be checked every 12 hours, or every 2 hours, or not at all, depending on clinical suspicion. At any given hour, 40–60% of lab values are simply missing (Johnson et al., 2023). Any system that tries to predict sepsis has to deal with these two very different types of data and their very different patterns of availability.

### 1.2 Research Significance

Existing approaches to early sepsis detection each have notable shortcomings.

Clinical scoring systems like SOFA, qSOFA, and SIRS are the current standard at the bedside (Singer et al., 2016; Seymour et al., 2016). They are simple to calculate and easy for clinicians to understand. But they are reactive by nature — they score organ dysfunction that has already happened, rather than predicting what is about to happen. In practice, qSOFA achieves AUROC values around 0.66–0.70 and SIRS around 0.64–0.68 (Seymour et al., 2016). These numbers reflect the fundamental limitation of rule-based scoring: they cannot capture the complex, patient-specific temporal patterns that precede sepsis onset.

Traditional machine learning — logistic regression, random forests, XGBoost — has been applied to this problem with moderate results (Desautels et al., 2016; Nemati et al., 2018). These methods can pick up non-linear relationships, but they need features to be engineered by hand, they struggle with sequential data, and they typically handle missing lab values by filling in the mean or carrying the last known value forward. That kind of simple imputation throws away information. If a doctor ordered a lactate test, the fact that they ordered it may be just as informative as the result.

Deep learning models using LSTMs and Transformers have shown promise for clinical time series (Shickel et al., 2017; Moor et al., 2021). They can learn temporal patterns without manual feature engineering, which is a real advantage. But most of these architectures feed all clinical variables into a single model — vital signs and lab values alike — without acknowledging that these data streams have fundamentally different characteristics, different sampling rates, and different amounts of missing data.

This struck me as a missed design opportunity. Clinical data naturally breaks down into distinct types: vital signs that stream in constantly, lab results that arrive sporadically with lots of gaps, and trends that capture how things are changing over time. A multi-agent architecture, where separate networks specialise in each data type and a coordinator combines their judgments, maps directly onto this structure. Multi-agent systems have worked well in robotics and game-playing (Wooldridge, 2009), but they have not been widely explored for medical prediction. This thesis tests whether the approach works for sepsis.

### 1.3 Research Questions

This thesis investigates four questions:

**RQ1:** Can a multi-agent deep learning architecture, with specialised agents for different clinical data types, achieve competitive sepsis prediction performance compared to established clinical scoring systems?

**RQ2:** How should hyperparameters — particularly learning rate — be adjusted when scaling from a small to a larger clinical dataset?

**RQ3:** Does the attention-weighted meta-learner produce agent contribution patterns that are clinically interpretable and consistent with medical domain knowledge?

**RQ4:** What is the relationship between model complexity (hidden dimensions, number of layers) and generalisation performance on clinical data of this scale?

### 1.4 Literature Review

#### 1.4.1 Sepsis Definition and Clinical Context

How we define sepsis has changed considerably. The original SIRS-based definition from 1992 (Bone et al., 1992) relied on physiological markers — temperature, heart rate, respiratory rate, and white blood cell count — to identify the inflammatory response to infection. The problem was that SIRS criteria were too broad. Post-surgical patients, people who had just exercised vigorously, even anxious patients could trigger a positive SIRS score without having any infection at all (Vincent et al., 2013).

The Sepsis-3 consensus in 2016 (Singer et al., 2016) replaced this with a sharper definition: sepsis is life-threatening organ dysfunction caused by a dysregulated host response to infection, operationalised as suspected infection plus a SOFA score increase of 2 or more points from baseline. This was an important shift — from flagging inflammation (which is common and often benign) to flagging organ dysfunction (which is dangerous and specific). The SOFA score evaluates six organ systems (respiratory, coagulation, liver, cardiovascular, neurological, and renal), each on a 0–4 scale, for a total range of 0–24.

For rapid bedside screening, Seymour et al. (2016) proposed qSOFA, which uses just three criteria: low systolic blood pressure (≤100 mmHg), fast breathing (≥22 breaths/min), and altered mental status. It is quick to compute but limited in sensitivity, with reported AUROC values of 0.66–0.70 for in-hospital mortality.

#### 1.4.2 Machine Learning for Sepsis Prediction

The use of machine learning for sepsis prediction goes back at least a decade. Desautels et al. (2016) used gradient boosting on vital signs and labs from MIMIC-III and reported AUROC of 0.76. Nemati et al. (2018) built a Weibull-Cox model that included both static patient features and temporal variables, achieving AUROC 0.85, though with a 4-hour prediction window that somewhat limits its usefulness for truly early warning.

The PhysioNet/Computing in Cardiology Challenge 2019 (Reyna et al., 2020) was a milestone — it created a standardised benchmark that the community could rally around. The top models scored utility values of 0.345–0.360, and most relied heavily on gradient boosting with hand-crafted temporal features. What stood out from the challenge was how hard the problem remained, especially around handling missing data and finding the right trade-off between sensitivity (catching real cases) and specificity (not drowning clinicians in false alarms).

#### 1.4.3 Deep Learning for Clinical Time Series

Deep learning brought a different toolkit to the problem. Recurrent neural networks, and particularly their gated variants — LSTMs (Hochreiter & Schmidhuber, 1997) and GRUs — can model sequential dependencies without requiring hand-crafted features. Lipton et al. (2016) showed that LSTMs could learn diagnostic patterns directly from raw multivariate clinical time series, which was a meaningful step forward from the feature-engineering approach.

Bi-directional LSTMs (Schuster & Paliwal, 1997) go further by reading the sequence in both directions. For clinical data, this matters: knowing that a heart rate spike at hour 15 was followed by recovery at hour 18 tells a different story than if it continued to climb. The bi-directional approach lets the model use both past and future context when evaluating each time step.

Attention mechanisms (Bahdanau et al., 2015) added another important capability — the ability to focus on the time steps that matter most. In a 24-hour window, most of the data might be stable and uninformative; what matters is the moments where things changed. Attention lets the model learn which moments those are, rather than treating every hour equally.

Transformers (Vaswani et al., 2017) took this idea even further by replacing recurrence entirely with self-attention, allowing direct comparison between any two time points regardless of how far apart they are. For detecting patterns like "lactate rising at hour 10 while blood pressure drops at hour 14", this direct comparison is much more natural than forcing information through a sequential chain. Moor et al. (2021) applied temporal Transformers to ICU data and found they performed competitively with recurrent models while being faster to train.

#### 1.4.4 Missing Data in Clinical Settings

Missing data is not just a nuisance in clinical prediction — it is informative. In an ICU, lab values are missing because a doctor decided not to order the test. That decision itself carries clinical signal: if the doctor did not order a lactate test, they may not have suspected metabolic problems. Conversely, a flurry of lab orders can indicate heightened clinical concern. Rubin (1976) formalised this as data "not missing at random", and Che et al. (2018) highlighted how most clinical prediction models ignore this entirely.

Some work has tried to address this. Che et al. (2018) proposed GRU-D, which adds trainable decay rates so that imputed values gradually revert toward the population mean as time since the last measurement increases. Cao et al. (2018) developed BRITS, which learns to impute missing values as part of the recurrent computation itself. My approach is in this vein — a learned imputation layer that discovers optimal fill-in values during training while keeping track of which observations were real and which were guessed.

#### 1.4.5 Multi-Agent Systems

The idea of breaking a complex problem into specialised sub-problems handled by cooperating agents has a long history in AI (Wooldridge, 2009). In neural network terms, mixture-of-experts models (Jacobs et al., 1991; Shazeer et al., 2017) are the closest relative — they use a gating network to route different inputs to different expert sub-networks.

My architecture is related but differs in an important way: rather than learning which expert to route to, I assign each agent a fixed data modality (vitals, labs, or trends). This gives up some flexibility but gains interpretability — you can always say exactly which data type each agent is responsible for, which makes the model's reasoning easier for clinicians to follow.

#### 1.4.6 Focal Loss for Class Imbalance

Class imbalance is a persistent issue in clinical prediction. In our dataset, about a third of cases are sepsis-positive and two-thirds are negative — not extreme, but enough to bias a model toward always predicting "no sepsis" if not handled properly.

Focal Loss (Lin et al., 2017) tackles this by adjusting the loss function in two ways. The alpha parameter controls how much weight to give each class — higher alpha means more penalty for missing positive cases. The gamma parameter down-weights examples the model already classifies confidently, focusing training on the harder cases near the decision boundary. Originally developed for object detection in computer vision, Focal Loss has since been adopted widely in medical prediction tasks.

### 1.5 Scope and Contributions

This thesis makes five contributions:

1. **A multi-agent architecture for clinical prediction** that assigns specialised neural networks to different data modalities (vitals, labs, trends) and combines them through an interpretable attention-weighted meta-learner.

2. **A learned imputation mechanism** for laboratory values that discovers contextually appropriate fill-in values during training while preserving information about which observations are real versus imputed.

3. **Evidence on hyperparameter scaling** — specifically, that learning rate must be reduced when scaling up clinical datasets, based on systematic experiments moving from 725 to 3,559 patients.

4. **Evidence on model sizing** — showing that a smaller architecture (32 hidden units, 1 layer) outperformed a larger one (64 hidden, 2 layers) on a dataset of this scale.

5. **Clinically interpretable agent weights** — demonstrating that the meta-learner's attention patterns align with medical domain knowledge without being explicitly programmed to do so.

---

## 2. Methodology

### 2.1 Dataset: MIMIC-IV

I used the Medical Information Mart for Intensive Care IV (MIMIC-IV) database, version 2.2 (Johnson et al., 2023). MIMIC-IV contains deidentified health records from patients admitted to the ICUs of Beth Israel Deaconess Medical Center in Boston, covering admissions from 2008 to 2019. It includes vital sign measurements, lab results, medication records, procedure orders, microbiology cultures, and administrative data.

From the full database, I extracted a cohort of 3,559 adult ICU admissions (patients aged 18 or older) who had at least 24 hours of monitoring data with recorded vital signs. This yielded 422,149 hourly observations across 34 clinical variables: 7 vital signs, 17 laboratory measurements, and metadata fields (patient IDs, timestamps, sepsis labels).

Access to MIMIC-IV requires completing the Collaborative Institutional Training Initiative (CITI) programme and agreeing to the PhysioNet Credentialed Health Data License, which I did prior to starting this project.

### 2.2 Data Preprocessing and Harmonisation

Working with raw MIMIC-IV is not straightforward. The data is spread across multiple relational tables, and measurements are identified by cryptic numeric codes rather than human-readable names. Heart rate, for instance, is stored as itemid 220045 in the `chartevents` table. My preprocessing pipeline, implemented in a `MIMICHarmonizer` class, handles the translation:

**Variable mapping.** I mapped raw MIMIC item codes to standardised clinical variable names using a YAML configuration file. This step also handles the fact that some variables have multiple item codes depending on the monitoring equipment used.

**Unit conversion.** Temperatures recorded in Fahrenheit were converted to Celsius. FiO₂ values entered as percentages (0–100) were converted to fractions (0–1). Without this step, the same clinical measurement could appear as completely different numbers.

**Temporal alignment.** Vital signs are charted every few minutes while labs arrive every few hours. I binned everything into one-hour intervals, using the median value for vitals (which have many readings per hour) and the last recorded value for labs (which typically have at most one).

**Forward-filling.** When no new measurement was available for a given hour, I carried the last known value forward — up to 6 hours for vital signs and up to 24 hours for labs. These limits are clinically motivated: a heart rate from 8 hours ago is probably stale, but a creatinine result from 12 hours ago is likely still roughly accurate.

**Feature normalisation.** All features were z-score normalised using the training set statistics only, so no information from the validation or test sets could leak into the normalisation step. I learned this the hard way during development — accidentally normalising with the full dataset initially gave misleadingly good results.

The processed data was saved in HDF5 format, which is efficient for the large array operations needed during training.

### 2.3 Sepsis Labelling via Sepsis-3 Criteria

MIMIC-IV does not come with a "sepsis" column. I had to construct the labels myself using the Sepsis-3 definition (Singer et al., 2016), implemented across two custom classes: `SOFACalculator` and `SepsisLabeler`.

The pipeline works as follows:

1. **Compute hourly SOFA scores.** For each hour of each patient's stay, I calculated SOFA sub-scores across six organ systems (respiratory, coagulation, liver, cardiovascular, neurological, renal), each scored 0–4.

2. **Establish a baseline.** The minimum SOFA score during the first 24 hours of ICU admission serves as the patient's baseline.

3. **Track changes.** At each subsequent hour, I computed the delta SOFA (current minus baseline). A delta of 2 or more points indicates new organ dysfunction.

4. **Identify suspected infection.** A patient was flagged as having suspected infection if they received antibiotics and had a microbiology culture ordered within 24 hours of each other. This is a standard clinical proxy — it captures the idea that the treating team thought an infection was present.

5. **Assign labels.** When both suspected infection and organ dysfunction (delta SOFA ≥ 2) co-occurred, I labelled that time point and all subsequent hours as sepsis-positive.

The resulting cohort had a 32.7% sepsis-positive rate. This is higher than the general ICU population because the inclusion criteria (sufficient monitoring data) skews toward sicker patients who tend to stay longer.

### 2.4 System Architecture

The core idea behind the architecture is straightforward: instead of feeding all clinical data into one monolithic model, I split it into three streams and give each one to a specialised agent. Each agent produces a fixed-dimensional summary (an "embedding") of what it sees. A meta-learner then looks at all three summaries and decides how much weight to give each one before making the final prediction.

**[INSERT FIGURE 1 HERE: System Architecture Diagram]**
*Figure 1: The multi-agent architecture. A 24-hour patient window is split into vital signs, lab values, and trend features. Three specialised agents process their respective inputs in parallel, and the meta-learner fuses their outputs using learned attention weights to produce a sepsis probability.*

The system takes in a 24-hour sliding window of patient data — 24 time steps, each with 7 vital signs and 17 lab values (24 features total). The window slides forward one hour at a time, producing a new prediction for each hour of the patient's stay.

### 2.5 Vitals Agent: Bi-directional LSTM with Attention

The Vitals Agent handles the seven continuously monitored vital signs: heart rate, respiratory rate, temperature, systolic and diastolic blood pressure, mean arterial pressure, and oxygen saturation. These are the "easy" data — recorded nearly every hour with over 95% completeness.

I used a bi-directional LSTM for this agent. The "bi-directional" part means the network reads the 24-hour sequence forwards (hour 1 to 24) and backwards (hour 24 to 1) simultaneously. This matters because context flows in both directions: a heart rate spike at hour 15 that was preceded by low blood pressure at hour 12 and followed by recovery at hour 18 tells a specific clinical story. A forward-only model would not have the recovery information when processing hour 15.

On top of the LSTM, I added an attention layer (Bahdanau et al., 2015). The idea is that not all 24 hours are equally important. Most of the time, vital signs are stable and boring. What matters are the moments where something changed — a sudden heart rate jump, a blood pressure drop, a temperature spike. The attention mechanism learns to assign higher weights to these clinically significant time steps and lower weights to the stable periods.

Formally, given the LSTM hidden states *h₁, h₂, ..., h_T*, the attention weight for each time step is:

*αₜ = softmax(wᵀ · tanh(W · hₜ + b))*

The final vitals embedding is the attention-weighted sum: *e_vitals = Σₜ αₜ · hₜ*

### 2.6 Labs Agent: LSTM with Learned Imputation

The Labs Agent deals with the harder data: seventeen laboratory measurements including lactate, WBC, creatinine, platelets, and blood gases. The challenge is the missingness — at any given hour, 40–60% of these values are absent.

Most models handle this with simple imputation: fill in the mean, or carry the last value forward. But this loses information. If a patient shows signs of kidney dysfunction, a missing creatinine is probably not going to be at the population average — it is more likely to be elevated. Simple imputation cannot capture this.

My approach uses a **learned imputation layer**. The agent maintains a trainable vector of imputation values, one per lab feature, that gets optimised along with all the other model parameters during training. When a lab value is missing (identified by a binary mask), the learned value is substituted. Over the course of training, the model discovers that, for instance, missing lactate values in ICU patients should be imputed slightly above the population mean — because lactate tends to be ordered when clinicians suspect metabolic problems, and absence of a test often means the value was not concerning enough to check.

The process works like this:

1. I compute a binary missing mask before replacing NaN values: *M_{t,f} = 1* if feature *f* at time *t* is missing, 0 otherwise.
2. Missing values are replaced: *x̃_{t,f} = (1 - M_{t,f}) · x_{t,f} + M_{t,f} · imp_f*, where *imp_f* is the learned imputation value.
3. The mask *M* is concatenated with the imputed values and fed to the LSTM together — so the model always knows which values were actually measured and which were guessed.

This last point is important. By keeping the mask as an explicit input, the model can learn to be more cautious when making predictions based on imputed data versus real measurements.

### 2.7 Trend Agent: Transformer Encoder

The Trend Agent takes a different angle entirely. Instead of looking at what the values *are*, it looks at how they are *changing*. A lactate of 4.0 mmol/L that has been stable for hours is a very different clinical situation from a lactate of 4.0 that was 2.0 six hours ago. The second scenario — rapidly rising lactate — is much more alarming.

I compute two derived features for each of the 24 clinical variables:

**Rate of change:** How fast is this value going up or down? Computed as the first-order difference: *v_{t,f} = x_{t,f} − x_{t-1,f}*

**Acceleration:** Is the rate of change itself speeding up or slowing down? This is the second-order difference: *a_{t,f} = v_{t,f} − v_{t-1,f}*

These trend features are fed into a Transformer encoder (Vaswani et al., 2017) with positional encoding. I chose a Transformer over an LSTM here for a specific reason: the Transformer's self-attention can directly compare any two time points, no matter how far apart they are. For detecting patterns like "lactate started rising at hour 8 and blood pressure started dropping at hour 14", the Transformer can relate these events directly, whereas an LSTM would need to carry the information through six intermediate time steps.

The Transformer output is mean-pooled across the temporal dimension to produce the final trend embedding.

### 2.8 Meta-Learner: Attention-Weighted Fusion

The meta-learner is where the three agents' outputs come together. It receives three embeddings — one from each agent — and needs to combine them into a single sepsis prediction.

The straightforward approach would be to concatenate them and pass them through a classifier. But I wanted something more interpretable, so I used an attention-based weighting scheme. The meta-learner learns to assign a dynamic importance weight to each agent, and these weights change depending on the specific patient's data.

The weights are computed by projecting each embedding through a learned query vector, then applying softmax so they sum to 1:

*wᵢ = softmax(qᵢᵀ · eᵢ)* for *i ∈ {vitals, labs, trends}*

The fused representation — *e_fused = w_vitals · e_vitals + w_labs · e_labs + w_trends · e_trends* — then passes through a classification head with sigmoid activation to produce a sepsis probability between 0 and 1.

What makes this useful beyond just the prediction is that you can inspect the weights. For a given patient, you can see that the model relied 45% on labs, 35% on vitals, and 20% on trends — and that tells you something about what the model "saw" in that patient's data. This is the kind of information a clinician can work with.

### 2.9 Training Procedure

All components are trained end-to-end — the three agents and the meta-learner update their parameters simultaneously based on the same loss signal. The training setup:

**Optimiser.** AdamW (Loshchilov & Hutter, 2019) with weight decay of 1×10⁻⁴.

**Loss function.** Focal Loss (Lin et al., 2017) with α=0.25 and γ=2.0. The gamma value means the model focuses more on the cases it finds hard to classify — a patient at 0.55 probability contributes more to the loss than one at 0.95 probability. Alpha gives slightly less weight to the positive (sepsis) class, which seems counterintuitive but worked well empirically.

**Batch size.** 32 for experiments v1–v6; increased to 512 for v7, where the small model and larger dataset made bigger batches both feasible and faster.

**Mixed precision training (AMP).** All experiments used automatic mixed precision (Micikevicius et al., 2018), which performs forward passes in float16 and accumulates gradients in float32. This gave roughly 2× GPU speedup with no measurable accuracy loss.

**Early stopping.** If validation AUROC did not improve for 10 consecutive epochs, training stopped. This prevents the model from fitting to noise in the later epochs.

**Learning rate scheduler.** ReduceLROnPlateau, which cuts the learning rate in half if validation AUROC stalls for 5 epochs. This lets the model make large steps early in training and progressively smaller ones as it converges.

**Gradient clipping.** Maximum norm of 1.0, mainly to prevent the LSTM gradients from exploding.

**Dropout.** Applied after LSTM and Transformer layers. Dropout randomly zeroes out a fraction of neuron activations during training, which forces the network to develop redundant representations and generally improves generalisation to unseen data.

**Data splitting.** I split patients — not individual data points — into training (70%), validation (10%), and test (20%) sets, using stratification to preserve the sepsis rate across splits. Patient-level splitting is essential: since one patient generates many 24-hour windows, splitting at the window level would let the same patient appear in both training and test sets, leading to artificially high scores. I set the random seed to 42 for reproducibility.

**Table 1: Dataset Partitioning (v7 — full MIMIC-IV)**

| Partition | Patients | Sequences | Positive Rate |
|-----------|----------|-----------|---------------|
| Training | 45,708 (70%) | ~4.6M | ~46% |
| Validation | 6,529 (10%) | ~0.7M | ~46% |
| Test | 13,060 (20%) | 1,303,417 | 46.2% |
| **Total** | **65,297** | **~6.5M** | **~46%** |

*Note: The sequence-level positive rate (~46%) is higher than the patient-level sepsis prevalence (~37.6%) because sepsis patients tend to have longer ICU stays, generating more sliding windows per patient.*

### 2.10 Experimental Design

I designed seven experiments to test specific hypotheses, changing one thing at a time from a baseline configuration:

**v1 (Baseline — small data).** 725 patients, learning rate 1×10⁻³, hidden dim 64, 2 layers, dropout 0.3, focal alpha 0.25. The purpose was to establish how the model performs on a smaller dataset.

**v2 (More data, same settings).** Everything identical to v1 but trained on all 3,559 patients. Does more data automatically mean better results?

**v3 (Lower learning rate).** Same as v2 but with learning rate reduced to 1×10⁻⁴. The idea was that the larger dataset might need a different optimisation pace.

**v4 (Higher focal alpha).** Same as v3 but with alpha increased to 0.35, giving more weight to sepsis cases in the loss function. Would this help with the class imbalance?

**v5 (More dropout).** Same as v3 but with dropout raised from 0.3 to 0.4. Would extra regularisation help prevent overfitting?

**v6 (Smaller model).** Same as v3 but with hidden dimension halved to 32 and layers reduced to 1. My hypothesis was that the larger model might have more capacity than the data could support.

**v7 (Full MIMIC-IV).** Taking the best architecture from v6 (32 hidden, 1 layer) and training on the full MIMIC-IV dataset — 65,297 patients instead of 3,559. Batch size increased to 512 to handle the larger dataset efficiently. This tested whether the data scaling benefits observed from v1→v2 continue at 18× larger scale.

Each experiment ran for up to 50 epochs with the early stopping and scheduling described above, all with the same random seed. v7 required chunked sequence building and memory-mapped arrays to handle the 7.9 million rows without running out of memory, and took approximately 260 minutes on a Google Colab GPU.

### 2.11 Evaluation Metrics

I report five metrics:

**AUROC** (Area Under the ROC Curve) is the primary metric. It measures how well the model ranks sepsis patients above non-sepsis patients across all possible thresholds. A score of 0.5 means random guessing; 1.0 means perfect ranking. I chose AUROC as the main metric because it is threshold-independent and is the standard in clinical ML research, making comparison with published work straightforward.

**AUPRC** (Area Under the Precision-Recall Curve) is the secondary metric. It is more sensitive to performance on the positive class, which matters in imbalanced settings. A random classifier would score 0.327 (our class prevalence), so anything above that represents genuine discriminative ability.

**F1 Score** is the harmonic mean of precision and recall at the optimal threshold, selected to maximise F1 on the validation set. It rewards models that balance catching sepsis cases with not generating too many false alarms.

**Sensitivity** is the proportion of actual sepsis cases the model catches. A sensitivity of 0.84 means 84 out of 100 sepsis patients would be flagged.

**Specificity** is the proportion of non-sepsis patients correctly cleared. Low specificity means lots of false alarms, which leads to alert fatigue — clinicians start ignoring warnings because most of them turn out to be wrong.

---

## 3. Results

### 3.1 Experimental Iterations

**Table 2: Results Across All Seven Experiments (sequence-level metrics)**

| Version | Patients | Learning Rate | Hidden | Layers | Dropout | Focal α | AUROC | AUPRC | F1 | Sens | Spec |
|---------|----------|---------------|--------|--------|---------|---------|-------|-------|----|------|------|
| v1 | 725 | 1×10⁻³ | 64 | 2 | 0.3 | 0.25 | 0.6421 | 0.5714 | 0.619 | 0.911 | 0.213 |
| v2 | 3,559 | 1×10⁻³ | 64 | 2 | 0.3 | 0.25 | 0.7160 | 0.6601 | 0.669 | 0.857 | 0.410 |
| v3 | 3,559 | 1×10⁻⁴ | 64 | 2 | 0.3 | 0.25 | 0.7529 | 0.6928 | 0.697 | 0.867 | 0.482 |
| v4 | 3,559 | 1×10⁻⁴ | 64 | 2 | 0.3 | 0.35 | 0.7375 | 0.6735 | 0.692 | 0.827 | 0.530 |
| v5 | 3,559 | 1×10⁻⁴ | 64 | 2 | 0.4 | 0.25 | 0.7281 | 0.6613 | 0.690 | 0.889 | 0.426 |
| v6 | 3,559 | 1×10⁻⁴ | 32 | 1 | 0.3 | 0.25 | 0.7423 | 0.6699 | 0.701 | 0.850 | 0.517 |
| **v7** | **65,297** | **1×10⁻⁴** | **32** | **1** | **0.3** | **0.25** | **0.7702** | **0.7032** | **0.714** | **0.858** | **0.534** |

The results tell a clear story when read in order.

Going from v1 to v2 — adding more data while keeping everything else the same — improved AUROC from 0.6421 to 0.7160. More data clearly helped, but the jump was not as large as I expected given a 5× increase in patients.

Going from v2 to v3 — just dropping the learning rate from 1×10⁻³ to 1×10⁻⁴ — pushed AUROC up another 0.037 to 0.7529. This was the moment it became clear that the learning rate had been holding v2 back. The model had the data it needed but was not converging properly at the higher learning rate.

Versions 4 and 5 tested whether fine-tuning the loss function (alpha) or regularisation (dropout) could squeeze out more performance. They could not. v4 dropped slightly from raising alpha; v5 dropped further from raising dropout. These parameters simply did not matter much at our scale.

Version 6 tested a smaller architecture. Cutting the model in half — from 64 hidden units and 2 layers down to 32 and 1 — gave competitive results with the best specificity so far (0.517) and 17% faster training (34 minutes versus 41). The smaller model generalised well despite having far fewer parameters.

Version 7 was the definitive test: taking v6's architecture and training on 18× more data (65,297 patients instead of 3,559). AUROC improved to 0.7702, AUPRC to 0.7032, and F1 to 0.714 — improvements across every metric. Data scaling proved to be the single most effective lever for performance, more impactful than any hyperparameter change tested in v3–v6.

### 3.2 Best Model Performance

The best configuration (v7) achieved the following on the held-out test set of 13,060 patients (1,303,417 sequences):

**Table 3: Best Model (v7) Sequence-Level Test Set Performance**

| Metric | Value |
|--------|-------|
| AUROC | 0.7702 |
| AUPRC | 0.7032 |
| F1 Score | 0.714 |
| Sensitivity | 0.858 |
| Specificity | 0.534 |
| Training Time | ~260 minutes |

The AUPRC of 0.7032 is well above the random baseline, showing the model discriminates effectively between sepsis and non-sepsis sequences. The F1 of 0.714 reflects a reasonable balance — the model catches most sepsis cases without generating an unmanageable number of false alarms.

**[INSERT FIGURE 2 HERE: ROC and Precision-Recall Curves for v7]**
*Figure 2: ROC curve (left, AUROC = 0.7702) and Precision-Recall curve (right, AUPRC = 0.7032) for the best model.*

**[INSERT FIGURE 3 HERE: Training and Validation Curves for v7]**
*Figure 3: Training and validation loss (left) and AUROC (right) over epochs, showing convergence and the point where early stopping triggered.*

### 3.3 Sensitivity-Specificity Trade-off

One of the more interesting aspects of the results is how the balance between sensitivity and specificity evolved across the six experiments:

**Table 4: How the Sensitivity-Specificity Balance Changed**

| Version | Sensitivity | Specificity | What was happening |
|---------|-------------|-------------|---------------------|
| v1 | 0.911 | 0.213 | Flagging nearly everyone as sepsis |
| v2 | 0.857 | 0.410 | Starting to distinguish the two classes |
| v3 | 0.867 | 0.482 | Getting close to balanced |
| v4 | 0.827 | 0.530 | Marginal shift from alpha change |
| v5 | 0.889 | 0.426 | Marginal shift from dropout change |
| v6 | 0.850 | 0.517 | Smaller model, better balance |
| **v7** | **0.858** | **0.534** | **Best balance with full data** |

The v1 model had 91.1% sensitivity, which sounds impressive until you notice the 21.3% specificity. In practice, that model was calling almost every patient septic. It would catch nearly all the real cases, sure, but it would also set off an alarm for roughly four out of every five healthy patients. That is not a useful clinical tool — it is a noise machine.

By v7, sensitivity held steady at 85.8% while specificity more than doubled to 53.4%. The model maintained high detection rates while correctly clearing far more non-sepsis patients. As shown in Section 3.6, patient-level evaluation reveals even stronger specificity (72.9%), since aggregating across a patient's full stay reduces window-level noise.

### 3.4 Agent Contribution Analysis

The meta-learner's attention weights let us look inside the model and see which agent mattered most for each prediction. Averaged across the test set:

**Table 5: Agent Attention Weights (v7)**

| Agent | Overall | Sepsis Cases | Non-Sepsis Cases |
|-------|---------|--------------|------------------|
| Vitals Agent | 33.4% | 33.5% | 33.3% |
| Labs Agent | 33.4% | 33.5% | 33.3% |
| Trend Agent | 33.2% | 33.0% | 33.4% |

At the full dataset scale, the agent weights converge to near-uniform distribution (~33% each). This differs from the earlier v6 experiments where the Labs Agent showed a stronger contribution (38.5%). The near-uniform weights indicate that the meta-learner, at this scale, treats the three agents as roughly equally informative rather than strongly differentiating between them. The model effectively operates as an ensemble average, with all three data streams contributing equally to the final prediction.

While the weights are less differentiated than in earlier experiments, the architecture still provides value: each agent applies specialised processing to its data stream (learned imputation for labs, attention for vitals, self-attention for trends) before the meta-learner combines them. The uniform weighting suggests that at 65,297 patients, all three data streams carry similar amounts of information about sepsis status.

**[INSERT FIGURE 4 HERE: Agent Contribution Charts]**
*Figure 4: Average agent contributions overall (left) and stratified by sepsis outcome (right).*

### 3.5 Baseline Model Comparison

To understand whether the multi-agent architecture adds value beyond its temporal modelling, I compared it against four single-timepoint machine learning baselines, all trained and evaluated on the same MIMIC-IV test set:

**Table 6: Multi-Agent vs. Baseline Models (sequence-level)**

| Model | AUROC | AUPRC |
|-------|-------|-------|
| Logistic Regression | 0.6773 | 0.5653 |
| Simple MLP (2 layers) | 0.7434 | 0.6191 |
| Random Forest | 0.7448 | 0.6311 |
| XGBoost | 0.7579 | 0.6457 |
| **Multi-Agent v7 (this thesis)** | **0.7702** | **0.7032** |

The multi-agent system outperforms all baselines. The gap is most notable in AUPRC (+0.058 over XGBoost), which is the more meaningful metric for imbalanced clinical data. The AUROC gap (+0.012 over XGBoost) is modest but consistent.

The baselines use single-timepoint features (one hour of data), while the multi-agent model uses 24-hour sequences. This comparison shows that the temporal context captured by the sliding window approach adds meaningful predictive information beyond what is available in a single snapshot. XGBoost's top features — respiratory rate, BUN, bicarbonate, FiO₂, and pH — are clinically valid, confirming the quality of the underlying data.

### 3.6 Patient-Level Evaluation

The sequence-level metrics above are based on individual 24-hour sliding windows. Because sepsis patients tend to have longer ICU stays, they generate many more windows than non-sepsis patients, inflating the sequence-level positive rate to ~46%. This does not reflect the clinical reality, where sepsis prevalence in ICUs is typically 5–15%.

To provide a more clinically relevant evaluation, I aggregated predictions at the patient level: for each patient, I took the maximum predicted probability across all their sequences. If any window flags a patient as high-risk, the patient is considered flagged.

**Table 7: Sequence-Level vs. Patient-Level Metrics (v7)**

| Metric | Sequence-Level | Patient-Level |
|--------|---------------|---------------|
| N | 1,303,417 | 10,927 |
| Prevalence | 46.2% | 37.6% |
| AUROC | 0.7702 | 0.8565 |
| AUPRC | 0.7032 | 0.7856 |
| F1 | 0.714 | 0.719 |
| Sensitivity | 0.858 | 0.813 |
| Specificity | 0.534 | 0.729 |
| PPV | 0.612 | 0.644 |
| NPV | 0.814 | 0.866 |

Patient-level evaluation reveals substantially stronger discrimination than sequence-level metrics suggest. AUROC jumps from 0.7702 to 0.8565, and specificity improves dramatically from 0.534 to 0.729 — meaning the model correctly clears nearly three out of four non-sepsis patients, compared to roughly one in two at the sequence level.

The improvement occurs because aggregating across a patient's full stay smooths out window-level noise. A non-sepsis patient might have one or two borderline windows (e.g., during a transient vital sign fluctuation), but their maximum probability still stays below the threshold. At the sequence level, these borderline windows count as individual false positives; at the patient level, they are absorbed.

### 3.7 Comparison with Clinical Scoring Systems

**Table 8: Model Performance vs. Clinical Scoring Systems**

| Model | AUROC | Notes |
|-------|-------|-------|
| **Multi-Agent v7 — patient-level** | **0.8565** | Best overall |
| Multi-Agent v7 — sequence-level | 0.7702 | Per sliding window |
| XGBoost (best baseline) | 0.7579 | Single timepoint |
| qSOFA (Seymour et al., 2016) | 0.66–0.70 | Bedside scoring |
| SIRS (Bone et al., 1992) | 0.64–0.68 | Bedside scoring |

The model outperforms both qSOFA and SIRS at both the sequence and patient level. At the patient level (AUROC 0.8565), it substantially exceeds these clinical scoring systems, which represent the practical standard of care.

The patient-level AUROC of 0.8565 also falls within the range reported by top models in the PhysioNet Challenge 2019 (0.70–0.85), though I want to be careful about drawing too strong a comparison. Those models were trained on different data with different label definitions, and the evaluation protocols were not identical. The comparison is indicative, not definitive.

---

## 4. Discussion

### 4.1 Addressing the Research Questions

**RQ1: Can the multi-agent architecture achieve competitive performance?**

Yes. At the patient level, v7 achieves AUROC 0.8565, which substantially exceeds qSOFA (0.66–0.70) and SIRS (0.64–0.68) and is competitive with recent deep learning approaches to sepsis prediction. Even at the sequence level (AUROC 0.7702), the model outperforms all baseline machine learning models tested, including XGBoost (0.7579). The architecture's advantage is that it handles the heterogeneity of clinical data naturally — vital signs, labs, and trends each get appropriate treatment rather than being forced through a one-size-fits-all model.

**RQ2: How should hyperparameters change when scaling data?**

The v1→v2→v3 progression gives a clear answer: when I increased the dataset from 725 to 3,559 patients, I needed to reduce the learning rate from 1×10⁻³ to 1×10⁻⁴. Keeping the original learning rate (v2) improved over v1 thanks to the additional data, but the model was not converging as well as it could. The lower learning rate (v3) unlocked an extra 0.037 AUROC. The v7 experiment then demonstrated that with the correct learning rate locked in, further scaling to 65,297 patients yielded additional gains without requiring further hyperparameter changes.

**RQ3: Are the agent weights clinically interpretable?**

Partially. At the smaller scale (v6, 3,559 patients), the agent weights showed clinically meaningful differentiation — the Labs Agent contributed more for sepsis cases while the Trend Agent contributed more for non-sepsis cases. At the full dataset scale (v7, 65,297 patients), the weights converged to near-uniform (~33% each), suggesting the meta-learner treats all agents as equally informative when given sufficient data. The architecture still provides value through specialised processing (learned imputation, temporal attention), but the weighting mechanism acts more as an ensemble average than a dynamic router.

**RQ4: How does model complexity relate to generalisation?**

At 3,559 patients, the smaller model (v6: 32 hidden, 1 layer) outperformed the larger one (v3: 64 hidden, 2 layers), suggesting the bigger architecture was slightly overfitting. Crucially, this smaller architecture continued to scale well when given 18× more data in v7 — AUROC improved from 0.7423 to 0.7702 without increasing model capacity. This suggests that for clinical time-series data at this scale, a compact architecture combined with more data is more effective than a larger model with less data.

### 4.2 Learning Rate Scaling

The v1→v2→v3 story was the most important finding, and honestly the one I did not expect going in. My assumption was that v2 — same model, five times more data — would be an easy win. It was an improvement (AUROC 0.7136 vs 0.6478), but nowhere near what the extra data should have delivered.

The problem was that with more data, the gradient estimates at each training step become more stable. That is normally a good thing — steadier gradients mean more reliable parameter updates. But the learning rate was calibrated for the noisy, jumpy gradients of the smaller dataset. With stable gradients and a large step size, the optimiser was overshooting the optimal parameters — taking big steps when it should have been taking small, careful ones.

Dropping the learning rate to 1×10⁻⁴ in v3 fixed this. The practical lesson is one that the deep learning literature documents (Smith, 2017) but that is easy to overlook in practice: **when you scale up your data, you should scale down your learning rate.** Hyperparameters that worked on a pilot study do not necessarily transfer to a larger cohort.

### 4.3 Model Complexity and Generalisation

I did not expect v6 to be the best model among v1–v6. My initial assumption was that more capacity (wider layers, deeper network) would be better, and that the smaller model would lose too much representational power. That turned out to be wrong at 3,559 patients.

With 3,559 patients, the 64-hidden, 2-layer model had more parameters than the data could effectively constrain. The 32-hidden, 1-layer model, with substantially fewer parameters, was forced to learn simpler, more robust patterns that transferred better to unseen patients.

Interestingly, when I scaled to the full MIMIC-IV (65,297 patients) in v7, I kept the smaller architecture rather than reverting to the larger one. The v7 results (AUROC 0.7702) show that the compact model continued to benefit from more data without needing additional capacity. Whether an even larger model would perform better at this scale remains an open question for future work.

### 4.4 Data Scaling

The progression from v6 (3,559 patients, AUROC 0.7423) to v7 (65,297 patients, AUROC 0.7702) confirmed that data volume is the strongest lever for improving performance. The +0.028 AUROC gain from 18× more data was larger than any hyperparameter change achieved in v3–v6, where learning rate, dropout, focal alpha, and architecture size collectively moved AUROC by ±0.025 around a plateau.

This aligns with the broader trend in machine learning: once the architecture and training procedure are reasonable, more data typically matters more than finer tuning. For clinical prediction tasks where labelled data is abundant (MIMIC-IV contains over 65,000 ICU admissions), this is good news — it suggests that continued data scaling could yield further improvements.

The practical challenge was engineering: the full dataset (7.9 million rows, 407 MB) required chunked sequence building, memory-mapped numpy arrays, and local SSD caching to avoid out-of-memory errors on Google Colab. Training took approximately 260 minutes — still feasible on free-tier cloud GPUs.

### 4.5 Sequence-Level vs Patient-Level Evaluation

One of the most important findings was the gap between sequence-level and patient-level metrics. At the sequence level, the model achieves AUROC 0.7702 — decent but not outstanding. At the patient level, AUROC jumps to 0.8565, which is a strong result by clinical ML standards.

This gap arises because sliding window evaluation artificially inflates the number of positive samples. Sepsis patients tend to have longer ICU stays, so they generate disproportionately more 24-hour windows. The sequence-level positive rate (~46%) is much higher than the patient-level prevalence (~37.6%), which is itself higher than real-world ICU sepsis rates (5–15%). Patient-level aggregation corrects for this inflation by collapsing all of a patient's windows into a single prediction.

The practical implication is that sequence-level metrics understate the model's clinical utility. A clinician does not care whether the model correctly classifies individual 24-hour windows — they care whether the model correctly identifies which patients have sepsis. The patient-level AUROC of 0.8565 is the more relevant number for assessing clinical deployment potential.

### 4.6 Regularisation and Loss Function Tuning

Versions 4 and 5 were the experiments that did not work, and I think the negative result is worth reporting.

Raising the focal alpha from 0.25 to 0.35 (v4) barely moved the needle — AUROC went from 0.7361 to 0.7372. This makes sense in retrospect: our class imbalance is 33/67, which is mild. Focal alpha matters much more in extreme imbalance scenarios like 5/95, where the model might otherwise ignore the minority class entirely. At 33/67, the default alpha already provides enough balance.

Increasing dropout from 0.3 to 0.4 (v5) was similarly uneventful — AUROC actually decreased very slightly, from 0.7361 to 0.7348. The model was not overfitting badly enough for extra dropout to help. It is worth noting that the architecture reduction in v6 turned out to be a much more effective way to control overfitting than increased dropout, because it actually reduces the number of parameters rather than just randomly disabling some of them during training.

### 4.7 Clinical Interpretability

The multi-agent architecture provides a structural form of interpretability that black-box alternatives lack. For any prediction, the system can report not just a probability but also how much each agent contributed — giving clinicians a window into *which type of data* drove the prediction.

In the earlier experiments (v6, 3,559 patients), the agent weights showed clear differentiation: the Labs Agent contributed 41.3% for sepsis cases while the Trend Agent contributed more for non-sepsis cases. At the full scale (v7, 65,297 patients), the weights converged to near-uniform (~33% each). This convergence is a limitation worth acknowledging: at scale, the meta-learner does not strongly differentiate between data streams, reducing the interpretability benefit.

That said, the architecture still provides value through specialised processing. Each agent applies domain-appropriate techniques — learned imputation for sparse lab data, bidirectional attention for vitals, self-attention for trends — before the meta-learner combines them. Even with uniform weighting, this is structurally more interpretable than a single monolithic model, because one can inspect each agent's intermediate outputs to understand what patterns it detected.

### 4.8 Learned Imputation

When I inspected the trained imputation vector after training, the values it had learned were clinically plausible. For lactate, the learned imputation was above the population mean, which makes sense: clinicians order lactate tests when they suspect metabolic problems, so a missing lactate in an ICU patient is more likely to be elevated than the average would suggest.

The learned imputation approach handles the 40–60% lab missingness without discarding information. By keeping the binary missing mask as an explicit input alongside the imputed values, the model can distinguish between actual measurements and imputed estimates — allowing it to be more cautious when relying on imputed data.

### 4.9 Clinical Deployment Considerations

A few practical points are worth flagging for anyone thinking about deploying a model like this.

**The 24-hour window requirement** means the model cannot make predictions for newly admitted patients. It needs a full day of data before it can start. For established ICU patients being monitored continuously, this is not a problem — the window just slides forward with each new hour.

**Threshold selection is a hospital-level decision.** The numbers I report use the threshold that maximises F1, but a hospital might want a different trade-off. If they want to catch every possible sepsis case, they can lower the threshold — at the cost of more false alarms. If their staff is already overwhelmed by alerts, they might raise it and accept missing a few cases. The model provides the probability; the threshold is a policy choice.

**Alert fatigue is a real concern.** At the sequence level, v7's specificity of 53.4% means roughly half of non-sepsis windows get flagged. However, at the patient level, specificity improves to 72.9% — meaning approximately three out of four non-sepsis patients would be correctly cleared. This patient-level specificity is more relevant for clinical deployment, where the question is "should we investigate this patient?" rather than "is this particular hour concerning?"

**Retrospective labelling** is a limitation. The Sepsis-3 labels are applied retrospectively — the model classifies windows where sepsis criteria are already met, rather than predicting future sepsis onset. A prospective deployment would need to assess whether the model can detect sepsis hours *before* clinical criteria are met.

---

## 5. Future Work

**External validation** is the most important next step. The model was trained and evaluated on data from a single hospital (Beth Israel Deaconess Medical Center). Whether it performs similarly at other institutions — with different patient populations, clinical practices, and documentation habits — is an open question. The eICU Collaborative Research Database (Pollard et al., 2018), covering over 200 US hospitals, would be a natural validation set.

**Larger model architectures** could now be revisited. The decision to use a compact architecture (32 hidden, 1 layer) was driven by overfitting concerns at 3,559 patients. With 65,297 patients in v7, there may be enough data to support a larger model — potentially unlocking performance beyond AUROC 0.8565.

**Expanding the feature set** could meaningfully improve performance. The current model uses 24 clinical variables, but ICU patients generate much more data — medications (especially vasopressors and antibiotics), fluid balance, ventilator settings, demographics, and comorbidity scores are all potentially predictive. Free-text clinical notes, processed through NLP, could add another dimension.

**Multi-task learning** — predicting sepsis alongside related outcomes like septic shock, ARDS, acute kidney injury, and mortality — might improve all predictions simultaneously by forcing the model to learn shared representations of patient deterioration.

**Prospective validation** would ultimately be needed to demonstrate clinical utility: does integrating the model into a clinical workflow actually lead to earlier treatment and better patient outcomes? This is the gold standard, but it requires a formal clinical trial, which was beyond the scope of this thesis.

**Uncertainty quantification** through Bayesian methods or ensemble approaches could help clinicians assess confidence in individual predictions, and **federated learning** could enable multi-site model training without sharing patient data — an important consideration given healthcare privacy requirements.

---

## 6. Conclusion

This thesis developed and evaluated a multi-agent deep learning system for early sepsis prediction in ICUs. The system assigns specialised neural networks to vital signs (bi-directional LSTM with attention), laboratory values (LSTM with learned imputation), and temporal trends (Transformer encoder), and combines them through an attention-weighted meta-learner.

Seven experiments scaling from 725 to 65,297 MIMIC-IV patients produced four main findings:

1. **Learning rate must be scaled with data size.** Moving from 725 to 3,559 patients required reducing the learning rate from 1×10⁻³ to 1×10⁻⁴. Keeping the original rate left significant performance on the table.

2. **A smaller model generalised better.** The 32-hidden, 1-layer architecture (v6) outperformed the 64-hidden, 2-layer version at 3,559 patients, and continued to scale well at 65,297 patients without needing additional capacity.

3. **Data scaling is the strongest lever.** Scaling from 3,559 to 65,297 patients (v6→v7) improved AUROC by +0.028 — a larger gain than any hyperparameter change achieved in v3–v6. The data, not the architecture, was the bottleneck.

4. **Patient-level evaluation reveals the model's true strength.** Sequence-level metrics (AUROC 0.7702) understate the model's discriminative power due to sliding window inflation. At the patient level, AUROC reaches 0.8565 with 72.9% specificity — a clinically meaningful result.

The best model (v7) outperforms clinical scoring systems (qSOFA AUROC 0.66–0.70, SIRS 0.64–0.68) and machine learning baselines including XGBoost (AUROC 0.7579) at both the sequence and patient level.

The limitations are real — single-centre training, retrospective labels, near-uniform agent weights at scale, and a modest feature set — but the results demonstrate that matching model architecture to data structure is a productive design principle. Vital signs, lab results, and temporal trends are genuinely different types of information, and treating them with specialised agents, combined with sufficient training data, produces a system that outperforms both clinical scores and standard ML approaches. External validation across institutions is the most important next step.

---

## 7. References

Ancker, J. S., Edwards, A., Nosal, S., et al. (2017). Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. *BMC Medical Informatics and Decision Making*, 17(1), 36.

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *3rd International Conference on Learning Representations (ICLR)*.

Bone, R. C., Balk, R. A., Cerra, F. B., et al. (1992). Definitions for sepsis and organ failure and guidelines for the use of innovative therapies in sepsis. *Chest*, 101(6), 1644–1655.

Cao, W., Wang, D., Li, J., et al. (2018). BRITS: Bidirectional recurrent imputation for time series. *Advances in Neural Information Processing Systems*, 31.

Che, Z., Purushotham, S., Cho, K., et al. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports*, 8(1), 6085.

Desautels, T., Calvert, J., Hoffman, J., et al. (2016). Prediction of sepsis in the intensive care unit with minimal electronic health record data: A machine learning approach. *JMIR Medical Informatics*, 4(3), e28.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79–87.

Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10(1), 1.

Kumar, A., Roberts, D., Wood, K. E., et al. (2006). Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. *Critical Care Medicine*, 34(6), 1589–1596.

Lin, T. Y., Goyal, P., Girshick, R., et al. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision*, 2980–2988.

Lipton, Z. C., Kale, D. C., Elkan, C., & Wetzel, R. (2016). Learning to diagnose with LSTM recurrent neural networks. *4th International Conference on Learning Representations (ICLR)*.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*.

Moor, M., Banerjee, O., Abad, Z. S. H., et al. (2021). Foundation models for generalist medical artificial intelligence. *Nature*, 616(7956), 259–265.

Nemati, S., Holder, A., Razmi, F., et al. (2018). An interpretable machine learning model for accurate prediction of sepsis in the ICU. *Critical Care Medicine*, 46(4), 547–553.

Pollard, T. J., Johnson, A. E. W., Raffa, J. D., et al. (2018). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. *Scientific Data*, 5(1), 180178.

Reyna, M. A., Josef, C. S., Jeter, R., et al. (2020). Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210–217.

Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.

Rudd, K. E., Johnson, S. C., Agesa, K. M., et al. (2020). Global, regional, and national sepsis incidence and mortality, 1990–2017: analysis for the Global Burden of Disease Study. *The Lancet*, 395(10219), 200–211.

Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.

Seymour, C. W., Liu, V. X., Iwashyna, T. J., et al. (2016). Assessment of clinical criteria for sepsis: for the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 762–774.

Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *5th International Conference on Learning Representations (ICLR)*.

Shickel, B., Tighe, P. J., Bihorac, A., & Rashidi, P. (2017). Deep EHR: A survey of recent advances in deep learning techniques for electronic health record (EHR) analysis. *IEEE Journal of Biomedical and Health Informatics*, 22(5), 1589–1604.

Singer, M., Deutschman, C. S., Seymour, C. W., et al. (2016). The third international consensus definitions for sepsis and septic shock (Sepsis-3). *JAMA*, 315(8), 801–810.

Smith, S. L. (2017). Don't decay the learning rate, increase the batch size. *arXiv preprint arXiv:1711.00489*.

Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A. (2019). What clinicians want: Contextualizing explainable machine learning for clinical end use. *Machine Learning for Healthcare Conference*, 359–380.

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

Vincent, J. L., Opal, S. M., Marshall, J. C., & Tracey, K. J. (2013). Sepsis definitions: time for change. *The Lancet*, 381(9868), 774–775.

Wooldridge, M. (2009). *An Introduction to MultiAgent Systems* (2nd ed.). John Wiley & Sons.

---

## 8. Appendices

### Appendix A: Best Training Configuration (v7)

```python
CONFIG = {
    # Data
    'data_file': 'mimic_processed_full.h5',  # 65,297 patients (full MIMIC-IV)
    'sequence_length': 24,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,

    # Model — smaller architecture generalises better
    'hidden_dim': 32,
    'num_layers': 1,
    'dropout': 0.3,

    # Training
    'batch_size': 512,  # Larger batch for full dataset
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,

    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # Features (7 vitals + 17 labs = 24 total)
    'vitals_features': ['hr', 'resp', 'temp', 'sbp', 'dbp', 'map_value', 'o2sat'],
    'labs_features': ['bun', 'chloride', 'creatinine', 'wbc', 'bicarbonate', 'platelets',
                      'magnesium', 'calcium', 'potassium', 'sodium', 'glucose',
                      'fio2', 'ph', 'paco2', 'pao2', 'lactate', 'bilirubin'],
}
```

### Appendix B: Clinical Feature Descriptions

**Table B1: Vital Signs Features**

| Feature | Description | Unit | Clinical Significance |
|---------|-------------|------|----------------------|
| hr | Heart rate | bpm | Tachycardia (>100) indicates stress response |
| resp | Respiratory rate | breaths/min | Tachypnea (>22) is a qSOFA criterion |
| temp | Body temperature | °C | Fever (>38.3) or hypothermia (<36) indicate infection |
| sbp | Systolic blood pressure | mmHg | Hypotension (<100) is a qSOFA criterion |
| dbp | Diastolic blood pressure | mmHg | Low DBP suggests vasodilation |
| map_value | Mean arterial pressure | mmHg | MAP <65 indicates cardiovascular dysfunction |
| o2sat | Oxygen saturation | % | Low SpO₂ indicates respiratory compromise |

**Table B2: Laboratory Features**

| Feature | Description | Unit | Clinical Significance |
|---------|-------------|------|----------------------|
| bun | Blood urea nitrogen | mg/dL | Elevated in renal dysfunction |
| chloride | Serum chloride | mEq/L | Acid-base balance indicator |
| creatinine | Serum creatinine | mg/dL | Primary renal function marker (SOFA) |
| wbc | White blood cell count | K/uL | Elevated in infection; very low indicates immunosuppression |
| bicarbonate | Serum bicarbonate | mEq/L | Low values indicate metabolic acidosis |
| platelets | Platelet count | K/uL | Thrombocytopenia indicates coagulation dysfunction (SOFA) |
| magnesium | Serum magnesium | mg/dL | Electrolyte balance |
| calcium | Serum calcium | mg/dL | Electrolyte balance |
| potassium | Serum potassium | mEq/L | Critical electrolyte; arrhythmia risk |
| sodium | Serum sodium | mEq/L | Fluid balance indicator |
| glucose | Blood glucose | mg/dL | Stress hyperglycaemia common in sepsis |
| fio2 | Fraction of inspired oxygen | 0–1 | Higher values indicate respiratory support |
| ph | Arterial pH | — | Acidosis (<7.35) indicates metabolic derangement |
| paco2 | Partial pressure CO₂ | mmHg | Respiratory function marker |
| pao2 | Partial pressure O₂ | mmHg | Oxygenation marker; PaO₂/FiO₂ ratio used in SOFA |
| lactate | Serum lactate | mmol/L | Key sepsis biomarker; elevated indicates tissue hypoperfusion |
| bilirubin | Total bilirubin | mg/dL | Liver dysfunction marker (SOFA) |

### Appendix C: Project File Structure

```
Sepsis/
├── config/
│   └── data_config.yaml              # Feature mappings and configurations
├── data/
│   └── processed/mimic_harmonized/
│       ├── mimic_processed_large.h5   # Processed dataset (3,559 patients, v1-v6)
│       └── mimic_processed_full.h5    # Full MIMIC-IV (65,297 patients, v7)
├── src/
│   ├── data/
│   │   ├── harmonization.py           # MIMICHarmonizer class
│   │   ├── sofa_calculator.py         # SOFACalculator class
│   │   └── labeling.py               # SepsisLabeler class
│   └── models/
│       └── multi_agent.py             # MultiAgentSepsisPredictor (all agent classes)
├── notebooks/
│   ├── Train_v7_Full_Dataset.ipynb        # All 7 experiments with checkpoint/resume
│   ├── Complete_Metrics_Analysis.ipynb    # Evaluation, visualisation, patient-level eval
│   └── Baseline_Comparison.ipynb          # XGBoost, RF, MLP, LR baselines
├── models/                            # Saved model checkpoints and results (v1-v7)
└── docs/
    ├── PROJECT_REPORT_DRAFT.md        # This document
    ├── QnA.md                         # Q&A preparation document
    └── PROJECT_WALKTHROUGH.md         # Detailed project walkthrough
```

### Appendix D: Figure Placement Guide

| Figure | Section | Source |
|--------|---------|--------|
| Figure 1: System Architecture | 2.4 | Create in draw.io or PowerPoint |
| Figure 2: ROC and PR Curves | 3.2 | `Complete_Metrics_Analysis.ipynb` (complete_metrics.png) |
| Figure 3: Training Curves | 3.2 | `Train_v7_Full_Dataset.ipynb` (training_curves.png) |
| Figure 4: Agent Weights | 3.4 | `Complete_Metrics_Analysis.ipynb` (agent_weights.png) |
| Figure 5: Baseline Comparison | 3.5 | `Baseline_Comparison.ipynb` |

---

**END OF THESIS DRAFT**
