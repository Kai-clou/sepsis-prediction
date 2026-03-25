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
| Logistic Regression | 0.6773 | 0.5653 |
| Random Forest | 0.7448 | 0.6311 |
| XGBoost | 0.7579 | 0.6457 |
| **Multi-Agent (ours)** | **0.7689** | **0.7008** |

### vs. Clinical Scores (published benchmarks)

| Score | AUROC |
|-------|-------|
| SIRS (Bone et al., 1992) | 0.64-0.68 |
| qSOFA (Seymour et al., 2016) | 0.66-0.70 |
| **Multi-Agent — patient-level** | **0.8571** |

### Key Ablation Findings (10-experiment study)

- Most hyperparameters (dropout, focal loss, weight decay) barely matter at scale (+-0.002 AUROC)
- Model size and sequence length are the two levers that actually matter
- Compact model (32/1) outperforms larger model (64/2) on all metrics

## Data

- **Source:** MIMIC-IV v2.2 (Johnson et al., 2023) via PhysioNet
- **Cohort:** 65,297 adult ICU admissions, ~7.9 million hourly observations
- **Sepsis definition:** Sepsis-3 (suspected infection + SOFA increase >= 2)
- **Features:** 7 vital signs + 17 laboratory measurements = 24 clinical variables
- **Split:** 70% train / 10% val / 20% test (patient-level, stratified)

Data files are not included (requires [PhysioNet credentialed access](https://physionet.org/)).

## How to Run

Notebooks are designed to run on **Google Colab** with data stored in Google Drive.

1. Get MIMIC-IV access via [PhysioNet](https://physionet.org/)
2. Run preprocessing notebooks to generate `.h5` files
3. Run `Train_v7_Full_Dataset.ipynb` for ablation training (E1-E10)
4. Run `Complete_Metrics_Analysis.ipynb` for evaluation
5. Run `Baseline_Comparison.ipynb` for baseline comparison


