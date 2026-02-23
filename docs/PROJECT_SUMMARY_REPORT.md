# Early Sepsis Prediction Using Multi-Agent Deep Learning

**Student:** Dinh (Jason)
**Supervisor:** Maoying
**Date:** February 2026

---

## 1. Problem

Sepsis is a life-threatening organ dysfunction caused by a dysregulated immune response to infection. It accounts for 1 in 3 hospital deaths, and every hour of delayed treatment increases mortality by 7-8%. Early detection is critical but difficult because early symptoms are non-specific.

## 2. Objective

Build an AI system that predicts sepsis onset in ICU patients **up to 6 hours before** clinical recognition, using the Sepsis-3 definition (suspected infection + SOFA score increase >= 2).

## 3. Data

- **Source:** MIMIC-IV (MIT, Beth Israel Deaconess Medical Center)
- **Cohort:** 3,559 ICU patients, 422,149 hourly observations
- **Sepsis prevalence:** 32.7% (1,164 patients)
- **Features:** 7 vital signs + 17 laboratory values (24 total)

## 4. Approach: Multi-Agent Architecture

Instead of a single model, we use **three specialist neural networks** combined by a coordinator:

| Agent | Data Type | Architecture | Why |
|-------|-----------|-------------|-----|
| **Vitals Agent** | HR, BP, Temp, SpO2, RR (7 features) | Bi-LSTM + Attention | Frequent, mostly complete data |
| **Labs Agent** | Creatinine, Lactate, WBC, etc. (17 features) | LSTM + Learned Imputation | Sparse data (60-99% missing) |
| **Trend Agent** | All 24 features (rate of change) | Transformer Encoder | Long-range temporal patterns |
| **Meta-Learner** | Agent embeddings | Attention-weighted fusion | Combines specialist opinions |

**Total parameters:** 151,425

## 5. Results

### Primary Performance (Best Model - v3)

| Metric | Value |
|--------|-------|
| **AUROC** | **0.7263** |
| **AUPRC** | **0.6536** |
| Sensitivity | 0.9229 (catches 92% of sepsis) |
| Specificity | 0.3399 |
| PPV (Precision) | 0.5270 |
| NPV | 0.8469 |
| F1 Score | 0.6709 |

### vs. Clinical Scores (Published Benchmarks)

| Method | AUROC | Our Advantage |
|--------|-------|---------------|
| SIRS | 0.64-0.68 | +7-13% |
| qSOFA | 0.66-0.70 | +4-10% |
| MEWS | 0.67-0.72 | +1-8% |
| **Our Model** | **0.7263** | -- |

*Note: Clinical score AUROCs are from published literature on different patient populations and serve as approximate reference points, not direct same-data comparisons. The ML comparison below is a direct, fair comparison on identical data and splits.*

### vs. Traditional ML (Same Data, Same Split)

| Method | AUROC | AUPRC |
|--------|-------|-------|
| Logistic Regression | ~0.65 | ~0.50 |
| Random Forest | ~0.67 | ~0.53 |
| XGBoost (best baseline) | 0.6876 | 0.5480 |
| **Multi-Agent (ours)** | **0.7263** | **0.6536** |

**Key finding:** Multi-Agent outperforms XGBoost by +5.6% AUROC and +19.3% AUPRC.

## 6. Experiments Summary

| Version | Change | AUROC |
|---------|--------|-------|
| v1 | Baseline (725 patients, lr=1e-3) | 0.7391 |
| v2 | Scaled to 3,559 patients | 0.6743 |
| **v3** | **Lower learning rate (1e-4)** | **0.7263** |
| v5 | Higher dropout (0.4) | 0.7198 |
| v6 | Simpler model (32h, 1 layer) | 0.7204 |

**Lesson:** More data initially hurt performance; tuning learning rate resolved this.

## 7. Technical Stack

- **Language:** Python
- **Framework:** PyTorch
- **Data:** MIMIC-IV via PhysioNet
- **Training:** Google Colab (GPU)
- **Key libraries:** scikit-learn, pandas, NumPy, matplotlib

## 8. Limitations

- Retrospective study on single-centre data (Beth Israel, Boston)
- Not validated in real-time clinical setting
- Would need external validation on Australian hospital data
- Regulatory approval (TGA) required before deployment

## 9. Next Steps

1. External validation on multi-centre data
2. Feature importance analysis (which vitals/labs matter most)
3. Explore larger patient cohorts from MIMIC-IV (50,000+ available)
4. Investigate real-time inference pipeline design

## 10. References

1. Singer M, et al. *Sepsis-3 Definition.* JAMA. 2016;315(8):801-810.
2. Seymour CW, et al. *qSOFA Assessment.* JAMA. 2016;315(8):762-774.
3. Johnson A, et al. *MIMIC-IV (v2.0).* PhysioNet. 2022.
4. Lin TY, et al. *Focal Loss.* IEEE ICCV. 2017.
5. Kaji DA, et al. *Attention-based Deep Learning for ICU.* PLOS ONE. 2019.

---

*Prepared for supervisor review meeting, February 2026*
