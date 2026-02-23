# Sepsis Prediction - Multi-Agent Deep Learning

Predicting sepsis onset in ICU patients 6 hours early using a multi-agent neural network architecture, trained on MIMIC-IV data.

**University of Technology Sydney - Undergraduate Project (2026)**

## What This Does

Uses 24 hours of patient vital signs and lab results to predict whether a patient will develop sepsis in the next 6 hours. Three specialist neural networks each analyze different aspects of the data, and a meta-learner combines their predictions.

## Architecture

```
Patient Data (24h window)
        |
        +---> Vitals Agent (Bi-LSTM)    -- HR, BP, Temp, SpO2, RR
        |
        +---> Labs Agent (LSTM+Imputation) -- Creatinine, Lactate, WBC, etc.
        |
        +---> Trend Agent (Transformer)  -- Rate of change across all features
                        |
                    Meta-Learner (Attention)
                        |
                  Sepsis Probability
```

## Results

| Metric | Value |
|--------|-------|
| AUROC | 0.7263 |
| AUPRC | 0.6536 |
| Sensitivity | 92.3% |
| Specificity | 34.0% |
| F1 Score | 0.6709 |

### vs. Traditional ML (same data, same split)

| Model | AUROC | AUPRC |
|-------|-------|-------|
| Logistic Regression | ~0.65 | ~0.50 |
| Random Forest | ~0.67 | ~0.53 |
| XGBoost | 0.6876 | 0.5480 |
| **Multi-Agent (ours)** | **0.7263** | **0.6536** |

### vs. Clinical Scores (published benchmarks)

| Score | AUROC | Source |
|-------|-------|--------|
| SIRS | 0.64-0.68 | Singer 2016 |
| qSOFA | 0.66-0.70 | Seymour 2016 |
| MEWS | 0.67-0.72 | Subbe 2001 |
| **Multi-Agent (ours)** | **0.7263** | This project |

*Note: Clinical score AUROCs are from published literature on different patient populations and serve as approximate reference points, not direct same-data comparisons.*

## Project Structure

```
Sepsis/
├── src/
│   ├── models/multi_agent.py       # Model architecture
│   └── data/                       # Preprocessing (harmonization, SOFA, labeling)
├── notebooks/
│   ├── MIMIC_IV_Preprocessing_Batched.ipynb  # Data pipeline
│   ├── Train_MultiAgent_Model.ipynb          # Training
│   ├── Baseline_Comparison.ipynb             # XGBoost, RF, LR, MLP
│   └── Complete_Metrics_Analysis.ipynb       # Full evaluation
├── config/                         # Feature configs
├── docs/                           # Documentation
└── models/                         # Saved weights & results (not in repo)
```

## Data

- **Source:** MIMIC-IV (PhysioNet) - real de-identified ICU records
- **Cohort:** 3,559 patients, 422,149 hourly observations
- **Sepsis definition:** Sepsis-3 (suspected infection + SOFA increase >= 2)
- **Features:** 7 vitals + 17 lab values

Data files are not included in this repo (requires PhysioNet credentials).

## How to Run

Notebooks are designed to run on **Google Colab** with data stored in Google Drive.

1. Get MIMIC-IV access via [PhysioNet](https://physionet.org/)
2. Run `MIMIC_IV_Preprocessing_Batched.ipynb` to process raw data
3. Run `Train_MultiAgent_Model.ipynb` to train the model
4. Run `Complete_Metrics_Analysis.ipynb` for evaluation

## Tech Stack

Python, PyTorch, scikit-learn, pandas, Google Colab (GPU)

## References

- Singer M, et al. *Sepsis-3 Definition.* JAMA. 2016.
- Johnson A, et al. *MIMIC-IV.* PhysioNet. 2022.
- Lin TY, et al. *Focal Loss.* IEEE ICCV. 2017.
