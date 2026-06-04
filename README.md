# Multi-Agent Deep Learning System for Early Sepsis Prediction in ICU

**University of Technology Sydney - Honours Thesis (2026)**

Predicting sepsis onset in ICU patients using a multi-agent neural network architecture, trained and evaluated on the full MIMIC-IV dataset (65,297 patients).

## Architecture

- **Vitals Agent**: Bi-directional LSTM with attention for continuously monitored vital signs (95%+ complete)
- **Labs Agent**: LSTM with learned imputation for sparse laboratory values (40-60% missing)
- **Trend Agent**: Transformer encoder for temporal derivatives (rate of change + acceleration)
- **Meta-Learner**: Attention-weighted fusion combining all three agent embeddings

## Results

### Best Model (E2: 32 hidden, 1 layer)

| Metric | Sequence-Level | Patient-Level |
|--------|---------------|---------------|
| AUROC | 0.7689 | **0.8571** |
| AUPRC | 0.7008 | **0.7844** |
| F1 | 0.714 | 0.718 |
| Sensitivity | 0.867 | 0.776 |
| Specificity | 0.518 | **0.766** |

### vs. Baselines (same data, same split)

| Model | AUROC | AUPRC |
|-------|-------|-------|
| Logistic Regression (1 time step) | 0.6773 | 0.5653 |
| Random Forest (1 time step) | 0.7448 | 0.6311 |
| XGBoost (1 time step) | 0.7579 | 0.6457 |
| Vanilla LSTM (E17, 24h window, early fusion) | 0.7506 | 0.6796 |
| **Multi-Agent (ours, 24h window, late fusion)** | **0.7689** | **0.7008** |

The snapshot baselines isolate the value of temporal context; the vanilla LSTM isolates the value of the multi-agent decomposition over a single recurrent model on the same windowed input.

### vs. Clinical Scores (re-evaluated on the same MIMIC-IV test cohort)

| Score | Sequence-Level AUROC | Patient-Level AUROC |
|-------|----------------------|----------------------|
| SIRS | 0.5853 | 0.7333 |
| qSOFA | 0.6208 | 0.7323 |
| **Multi-Agent (ours)** | **0.7689** | **0.8571** |

### Key Ablation Findings (10-experiment hyperparameter study, E1-E10)

- Most hyperparameters (dropout, focal loss, learning rate, weight decay) barely matter at scale (within +-0.003 AUROC)
- Model size and sequence length are the two levers that actually matter
- Compact model (32/1) outperforms larger model (64/2) on AUROC, AUPRC, and specificity

### Per-Agent Ablation (E11-E16)

- **Labs Agent dominates**: a Labs-only model reaches AUROC 0.7618 (within 0.007 of the full ensemble); removing the Labs Agent costs 0.071 AUROC
- Vitals-only (0.6565) and Trend-only (0.6770) are individually weak but mutually substitutable
- Confirms the exploratory finding that laboratory features carry most of the predictive signal

### Calibration

- Raw model is poorly calibrated (ECE 0.1253, Brier 0.2193) - an expected consequence of focal-loss training, which optimises ranking over absolute probabilities
- Post-hoc isotonic regression (fit on one half of the test set, applied to the other) reduces ECE to 0.0014 and Brier to 0.1939, leaving AUROC unchanged (monotonic transform)

## Data

- **Source:** MIMIC-IV v2.2 (Johnson et al., 2023) via PhysioNet
- **Cohort:** 65,297 adult ICU admissions, ~7.9 million hourly observations
- **Sepsis definition:** Sepsis-3 (suspected infection + SOFA increase >= 2)
- **Features:** 7 vital signs + 17 laboratory measurements = 24 clinical variables
- **Split:** 70% train / 10% val / 20% test (patient-level, stratified)

Raw data is **not** included and is **not** redistributable: MIMIC-IV requires [PhysioNet credentialed access](https://physionet.org/) and CITI training under the PhysioNet Credentialed Health Data License. The preprocessing notebook regenerates the processed cohort end-to-end from the raw v2.2 download.

## Repository Structure

```
src/
  data/        harmonization, Sepsis-3 labelling, SOFA score
  models/      multi_agent.py (Vitals/Labs/Trend agents + meta-learner)
notebooks/     end-to-end pipeline (see below)
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `MIMIC_IV_Preprocessing_Batched.ipynb` | Build the cohort HDF5 from raw MIMIC-IV: Sepsis-3 labelling, hourly resampling, normalisation, 24h windows |
| `Data_Exploration.ipynb` | Missingness rates, feature distributions, feature-outcome correlations, pre-onset trajectories |
| `Train_v7_Full_Dataset.ipynb` | Train the multi-agent model: hyperparameter ablation (E1-E10) and per-agent ablation (E11-E16) |
| `Vanilla_LSTM_Baseline.ipynb` | Unified vanilla LSTM temporal baseline (E17, early-fusion test) |
| `Baseline_Comparison.ipynb` | Snapshot ML baselines: logistic regression, MLP, random forest, XGBoost |
| `Clinical_Score_Comparison.ipynb` | qSOFA and SIRS re-implemented on the same windows and labels |
| `Complete_Metrics_Analysis.ipynb` | Final metrics for E2: sequence- vs patient-level, threshold selection |
| `Calibration_Analysis.ipynb` | Reliability diagram, ECE, Brier, isotonic recalibration |

## How to Run

Notebooks are designed to run on **Google Colab** with data stored in Google Drive.

1. Get MIMIC-IV access via [PhysioNet](https://physionet.org/) and complete CITI training
2. Run `MIMIC_IV_Preprocessing_Batched.ipynb` to generate the processed `.h5` cohort
3. Run `Train_v7_Full_Dataset.ipynb` for the multi-agent ablations (E1-E16)
4. Run `Vanilla_LSTM_Baseline.ipynb`, `Baseline_Comparison.ipynb`, and `Clinical_Score_Comparison.ipynb` for the comparison models
5. Run `Complete_Metrics_Analysis.ipynb` and `Calibration_Analysis.ipynb` for evaluation and calibration

All experiments were run on Google Colab Pro with a single NVIDIA A100 GPU.
