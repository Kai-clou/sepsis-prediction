# Antigravity: Multi-Agent Sepsis Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A novel multi-agent deep learning architecture for predicting sepsis onset 6-12 hours before clinical recognition, trained on MIMIC-IV with external validation on PhysioNet Computing in Cardiology Challenge 2019 data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Innovation](#key-innovation)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Expected Results](#expected-results)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Antigravity** is an academic research project investigating whether a **multi-agent neural architecture** can outperform traditional single-model approaches for early sepsis prediction in intensive care units (ICUs).

### Problem Statement

Sepsis is a life-threatening condition affecting 49 million people annually, with 11 million deaths worldwide. Early detection (6-12 hours before clinical recognition) can significantly improve patient outcomes through timely intervention. However, sepsis prediction remains challenging due to:

- **Heterogeneous clinical data**: Irregular vital signs, sparse lab measurements, complex temporal patterns
- **Class imbalance**: Sepsis affects only 4-7% of ICU patients
- **Distribution shift**: Models trained on one hospital system often fail on others

### Proposed Solution

This project introduces a **multi-agent architecture** that:
1. Separates temporal and static feature processing into specialized agents
2. Fuses agent representations for final prediction
3. Uses time-to-event regression as auxiliary supervision
4. Validates generalization across two independent datasets

---

## 🚀 Key Innovation

### Multi-Agent vs. Traditional Single-Model Approaches

**Traditional Approach:** A single LSTM or XGBoost model processes all features together.

**Antigravity Approach:** Specialized agents handle different data modalities:

```
┌──────────────────────────────────────────────────────────────┐
│                    ANTIGRAVITY ARCHITECTURE                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐         ┌──────────────────┐           │
│  │  AGENT A        │         │   AGENT B        │           │
│  │  (Temporal)     │         │   (Static)       │           │
│  │                 │         │                  │           │
│  │  Bi-LSTM        │         │   FFN            │           │
│  │  - Vitals (HR)  │         │   - Age          │           │
│  │  - Labs (Lac)   │         │   - Sex          │           │
│  │  - 24h window   │         │   - Admission    │           │
│  └────────┬────────┘         └────────┬─────────┘           │
│           │                           │                      │
│           └──────────┬────────────────┘                      │
│                      │                                       │
│                ┌─────▼──────┐                                │
│                │   FUSION   │                                │
│                │   LAYER    │                                │
│                └─────┬──────┘                                │
│                      │                                       │
│           ┌──────────┴──────────┐                            │
│           │                     │                            │
│     ┌─────▼──────┐       ┌─────▼──────┐                     │
│     │ CLASSIFIER │       │  AGENT C   │                     │
│     │  (Sepsis)  │       │   (TTE)    │                     │
│     └────────────┘       └────────────┘                     │
│      Prediction          Time-to-Event                       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Advantages:**
- **Modularity**: Agents can be improved independently
- **Interpretability**: Agent attention reveals which features drive predictions
- **Transfer Learning**: Agents pre-trained on one dataset transfer to another

---

## 🏗️ Architecture

### Agent A: Temporal LSTM Agent
- **Input**: 24-hour window of vital signs and labs (40 variables, hourly samples)
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Output**: 256-dimensional temporal embedding

### Agent B: Static FFN Agent
- **Input**: Patient demographics (age, sex, admission type, ICU type)
- **Architecture**: 3-layer feed-forward network with batch normalization
- **Output**: 64-dimensional static embedding

### Fusion Layer
- **Input**: Concatenated Agent A + Agent B embeddings (320 dimensions)
- **Architecture**: Fully connected layer with ReLU + dropout
- **Output**: 128-dimensional fused representation

### Classification Head
- **Input**: Fused representation
- **Output**: Binary sepsis probability (sigmoid activation)

### Agent C: Time-to-Event Regression
- **Input**: Fused representation
- **Output**: Predicted hours until sepsis onset (for positive cases)
- **Loss**: Masked MSE (calculated only for sepsis-positive patients)

### Multi-Task Loss Function

```
L_total = α * L_classification + β * L_time_to_event

where:
  L_classification = Weighted Binary Cross-Entropy (handles class imbalance)
  L_time_to_event  = Masked MSE (ignores non-sepsis patients)
  α = 0.7 (primary task weight)
  β = 0.3 (auxiliary task weight)
```

---

## 📊 Datasets

### Primary Training: MIMIC-IV v2.2
- **Source**: MIT Laboratory for Computational Physiology
- **Description**: De-identified ICU data from Beth Israel Deaconess Medical Center (2008-2019)
- **Size**: ~70,000 ICU stays
- **Access**: Completed CITI training, credentialed via PhysioNet
- **Usage**: Train (70%), Validation (15%), Internal Test (15%)

**Key Tables Used:**
- `chartevents`: Vital signs (Heart Rate, Blood Pressure, Temperature, etc.)
- `labevents`: Laboratory measurements (Lactate, Creatinine, Bilirubin, Platelets)
- `prescriptions`: Antibiotic administration
- `microbiologyevents`: Culture orders
- `patients`, `admissions`, `icustays`: Demographics and admission details

### External Validation: PhysioNet/CinC Challenge 2019
- **Source**: Physionet Computing in Cardiology Challenge 2019
- **Description**: Multi-hospital ICU data (3 US hospital systems)
- **Size**: 40,336 patient records (60 GB)
- **Format**: Pipe-separated values (.psv) with 40 hourly variables + SepsisLabel
- **Usage**: External test set (100% - no training)

**Why External Validation Matters:**
- Tests model generalization across different hospital systems
- Different patient populations, clinical practices, equipment calibrations
- True test of clinical utility beyond fitting training distribution

---

## 💻 Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- 32GB+ RAM (for processing MIMIC-IV)
- ~200GB disk space (datasets + processed files)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/antigravity.git
cd antigravity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### Step 1: Download Data

```bash
# Download MIMIC-IV from Google Drive (requires credentials)
python scripts/01_download_data.py --dataset mimic_iv --output data/raw/mimic_iv/

# Download PhysioNet CinC 2019 (requires PhysioNet account)
python scripts/01_download_data.py --dataset cinc2019 --output data/raw/cinc2019/
```

### Step 2: Preprocess Data

```bash
# Process MIMIC-IV: Harmonization + SOFA + Sepsis-3 Labeling
python scripts/02_preprocess_mimic.py \
    --config config/data_config.yaml \
    --output data/processed/mimic_harmonized/

# Process PhysioNet CinC 2019
python scripts/03_preprocess_cinc.py \
    --config config/data_config.yaml \
    --output data/processed/cinc_processed/
```

**Expected Output:**
- `data/processed/mimic_harmonized/train_temporal.h5` (~20GB)
- `data/processed/mimic_harmonized/train_static.h5` (~500MB)
- `data/processed/mimic_harmonized/train_labels.h5` (~100MB)

### Step 3: Train Multi-Agent Model

```bash
# Train Antigravity multi-agent system
python scripts/04_train_multi_agent.py \
    --model_config config/model_config.yaml \
    --training_config config/training_config.yaml \
    --data data/processed/mimic_harmonized/ \
    --experiment_name "antigravity_v1"

# Monitor training with MLflow UI
mlflow ui --backend-store-uri ./mlruns
# Open browser: http://localhost:5000
```

### Step 4: Train Baseline Models

```bash
# Train comparison models
python scripts/05_train_baselines.py \
    --models single_lstm xgboost logistic_regression \
    --data data/processed/mimic_harmonized/
```

### Step 5: Evaluate Models

```bash
# Internal validation (MIMIC-IV held-out test set)
python scripts/06_evaluate_internal.py \
    --model results/checkpoints/multi_agent_best.pth \
    --data data/processed/mimic_harmonized/test/

# External validation (PhysioNet CinC 2019)
python scripts/07_evaluate_external.py \
    --model results/checkpoints/multi_agent_best.pth \
    --data data/processed/cinc_processed/test/
```

### Step 6: Generate Thesis Figures

```bash
# Create all plots and tables for thesis
python scripts/08_generate_thesis_plots.py \
    --experiments mlruns/ \
    --output results/figures/
```

---

## 📁 Project Structure

```
antigravity/
│
├── README.md                          # This file
├── IMPLEMENTATION_PLAN.md             # Detailed technical specification
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── config/                            # Configuration files
│   ├── data_config.yaml              # Data harmonization settings
│   ├── model_config.yaml             # Architecture hyperparameters
│   ├── training_config.yaml          # Training settings
│   └── evaluation_config.yaml        # Evaluation metrics config
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                          # Original datasets
│   │   ├── mimic_iv/                 # MIMIC-IV CSVs
│   │   └── cinc2019/                 # PhysioNet Challenge PSVs
│   ├── processed/                    # Harmonized datasets
│   │   ├── mimic_harmonized/         # MIMIC mapped to CinC schema
│   │   └── cinc_processed/           # Processed CinC data
│   └── metadata/                     # Data provenance logs
│
├── src/                               # Source code
│   ├── data/                         # ETL pipeline
│   │   ├── mimic_loader.py           # Load MIMIC-IV from storage
│   │   ├── cinc_loader.py            # Load PhysioNet CinC data
│   │   ├── harmonization.py          # Schema mapping & unit conversion
│   │   ├── sofa_calculator.py        # Sepsis-3 SOFA score implementation
│   │   ├── labeling.py               # Ground truth generation
│   │   ├── imputation.py             # Handle missing data
│   │   └── feature_engineering.py    # Create model inputs
│   │
│   ├── models/                       # Neural network architectures
│   │   ├── agent_a.py                # Temporal LSTM agent
│   │   ├── agent_b.py                # Static FFN agent
│   │   ├── agent_c.py                # Time-to-event regression head
│   │   ├── fusion.py                 # Agent fusion layer
│   │   ├── multi_agent_system.py     # Complete Antigravity model
│   │   └── baselines.py              # Single LSTM, XGBoost, etc.
│   │
│   ├── training/                     # Training orchestration
│   │   ├── trainer.py                # Main training loop
│   │   ├── loss_functions.py         # Multi-task loss
│   │   ├── metrics.py                # AUROC, Utility Score, etc.
│   │   └── callbacks.py              # Early stopping, checkpointing
│   │
│   ├── evaluation/                   # Model assessment
│   │   ├── internal_validation.py    # MIMIC-IV test evaluation
│   │   ├── external_validation.py    # CinC 2019 evaluation
│   │   └── utility_score.py          # PhysioNet utility metric
│   │
│   └── utils/                        # Helper functions
│       ├── config_loader.py          # YAML configuration parser
│       └── logging_utils.py          # Custom loggers
│
├── scripts/                           # Executable workflows
│   ├── 01_download_data.py           # Data acquisition from Google Drive
│   ├── 02_preprocess_mimic.py        # MIMIC-IV ETL pipeline
│   ├── 03_preprocess_cinc.py         # CinC 2019 processing
│   ├── 04_train_multi_agent.py       # Train Antigravity system
│   ├── 05_train_baselines.py         # Train comparison models
│   ├── 06_evaluate_internal.py       # Internal validation
│   ├── 07_evaluate_external.py       # External validation
│   └── 08_generate_thesis_plots.py   # Figure generation
│
├── notebooks/                         # Exploratory analysis
│   ├── 01_eda_mimic.ipynb            # Explore MIMIC-IV raw data
│   ├── 02_eda_cinc.ipynb             # Explore CinC 2019 data
│   ├── 03_sofa_validation.ipynb      # Validate SOFA calculations
│   └── 04_model_comparison.ipynb     # Compare model performance
│
├── tests/                             # Unit tests
│   ├── test_harmonization.py         # Test itemid mapping
│   ├── test_sofa_calculation.py      # Validate SOFA logic
│   └── test_models.py                # Test model architectures
│
├── docs/                              # Documentation
│   ├── methodology.md                # Detailed methods chapter
│   ├── data_dictionary.md            # Variable definitions
│   └── reproducibility_guide.md      # Reproduction instructions
│
└── results/                           # Generated outputs (gitignored)
    ├── figures/                      # Thesis plots
    ├── checkpoints/                  # Model weights
    └── predictions/                  # Model outputs
```

---

## 🔬 Methodology

### Data Harmonization

**Challenge**: MIMIC-IV uses `itemid` codes for variables, while CinC 2019 uses canonical variable names. Units and sampling frequencies differ.

**Solution**: Force MIMIC-IV to "speak" the CinC schema:

| CinC Variable | MIMIC-IV itemid | Source Table | Unit | Conversion |
|---------------|-----------------|--------------|------|------------|
| HR | 220045 | chartevents | bpm | None |
| Temp | 223761, 223762 | chartevents | °C | F→C: (F-32)×5/9 |
| SBP | 220179, 220050 | chartevents | mmHg | None |
| Lactate | 50813 | labevents | mmol/L | None |

**Temporal Alignment**: MIMIC-IV has irregular timestamps (every few minutes). We create hourly bins and aggregate:
- **Vitals**: Median (robust to outliers)
- **Labs**: Last observation (most recent value)

### Sepsis-3 Definition

**Ground truth** is generated using the official Sepsis-3 definition (Singer et al., JAMA 2016):

1. **Suspected Infection**: Co-occurrence of antibiotics + body fluid cultures within 24-hour window
2. **Organ Dysfunction**: SOFA score increase ≥2 from baseline (minimum in first 24h ICU)
3. **Sepsis Onset**: Earliest time of infection suspicion or organ dysfunction

### Prediction Window

**Labeling Strategy** (6-12 hour early prediction window):
- **Positive (1)**: Time steps 6-12 hours before sepsis onset
- **Negative (0)**: All data for non-septic patients; data >12h before onset for septic patients
- **Excluded**: Time steps <6h before onset (too close) and after onset (already septic)

**Rationale**:
- >12h predictions have too many false positives
- <6h predictions leave insufficient intervention time
- 6-12h window is clinically actionable

### SOFA Score Calculation

Sequential Organ Failure Assessment (SOFA) evaluates 6 organ systems:

1. **Respiratory**: PaO2/FiO2 ratio, ventilation status → 0-4 points
2. **Coagulation**: Platelet count → 0-4 points
3. **Liver**: Bilirubin level → 0-4 points
4. **Cardiovascular**: Mean arterial pressure, vasopressor doses → 0-4 points
5. **CNS**: Glasgow Coma Scale → 0-4 points
6. **Renal**: Creatinine, urine output → 0-4 points

**Total SOFA**: 0-24 points (higher = more organ dysfunction)

**Baseline SOFA**: Minimum score in first 24 hours of ICU admission (represents patient's "normal" state)

**Organ Dysfunction**: SOFA increase ≥2 from baseline

### Missing Data Handling

Clinical time-series are notoriously sparse. **Imputation strategy**:

1. **Forward Fill**: Carry last observation forward (up to 6h for vitals, 24h for labs)
2. **Mean Imputation**: Use population mean if no recent value exists
3. **Missingness Indicators**: Add binary flags (0=observed, 1=missing) as additional features

**Rationale**: Forward fill captures "last known state" assumption common in clinical practice.

### Class Imbalance

Sepsis prevalence in ICU: **4-7%** (highly imbalanced).

**Mitigation Strategies**:
1. **Weighted Loss**: `pos_weight=12` in Binary Cross-Entropy (penalizes false negatives 12× more)
2. **Oversampling**: Randomly duplicate minority class during training
3. **AUPRC Metric**: Area Under Precision-Recall Curve (more informative than AUROC for imbalanced data)

---

## 📈 Expected Results

### Target Performance Metrics

Based on literature review and preliminary experiments:

#### Internal Validation (MIMIC-IV Held-Out Test Set)

| Model | AUROC | AUPRC | Sensitivity@90%Spec | Utility Score |
|-------|-------|-------|---------------------|---------------|
| **Multi-Agent (Antigravity)** | **0.82-0.88** | **0.35-0.45** | **0.65-0.75** | **0.28-0.35** |
| Single LSTM | 0.78-0.84 | 0.30-0.40 | 0.60-0.70 | 0.22-0.28 |
| XGBoost | 0.75-0.82 | 0.25-0.35 | 0.55-0.65 | 0.18-0.25 |
| Logistic Regression | 0.70-0.76 | 0.20-0.30 | 0.45-0.55 | 0.12-0.18 |

#### External Validation (PhysioNet CinC 2019)

| Model | AUROC | AUPRC | Utility Score | Performance Drop |
|-------|-------|-------|---------------|------------------|
| **Multi-Agent (Antigravity)** | **0.75-0.82** | **0.28-0.38** | **>0.25** | **5-10%** |
| Single LSTM | 0.72-0.78 | 0.25-0.35 | 0.20-0.25 | 8-12% |

**Expected Findings**:
1. Multi-agent architecture outperforms single-model baselines by 3-5% AUROC
2. External validation shows 5-10% performance drop (acceptable domain shift)
3. Time-to-event auxiliary task improves calibration
4. Agent attention reveals interpretable patterns (e.g., lactate spikes, heart rate trends)

### PhysioNet Utility Score

Official challenge metric that rewards timely predictions and penalizes false alarms:

```
Utility = Σ(rewards for true positives) - Σ(penalties for false positives)

where:
  - True positive 6-12h early: +1.0
  - True positive 0-6h early: +0.5
  - False positive: -0.05
  - False negative: -1.0
```

**Target**: Utility Score >0.25 (top quartile of CinC 2019 submissions)

---

## 📚 References

### Sepsis-3 Definition
- Singer M, et al. **"The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)."** JAMA. 2016;315(8):801-810. [DOI](https://doi.org/10.1001/jama.2016.0287)

### MIMIC-IV Database
- Johnson A, et al. **"MIMIC-IV, a freely accessible electronic health record dataset."** Scientific Data. 2023;10:1. [DOI](https://doi.org/10.1038/s41597-022-01899-x)
- [MIMIC-IV Documentation](https://mimic.mit.edu/)

### PhysioNet Challenge 2019
- Reyna MA, et al. **"Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019."** Critical Care Medicine. 2020;48(2):210-217. [DOI](https://doi.org/10.1097/CCM.0000000000004145)
- [Challenge Website](https://physionet.org/content/challenge-2019/)

### SOFA Score Implementation
- Vincent JL, et al. **"The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure."** Intensive Care Med. 1996;22:707-710. [DOI](https://doi.org/10.1007/BF01709751)
- [OpenSep Pipeline](https://github.com/joamats/mit-sepsis) - Open-source SOFA calculation for MIMIC

### Multi-Agent Deep Learning
- Kamaleswaran R, et al. **"Applying Artificial Intelligence to Identify Physiomarkers Predicting Severe Sepsis in the PICU."** Pediatric Critical Care Medicine. 2018;19(10):e495-e503. [DOI](https://doi.org/10.1097/PCC.0000000000001666)

---

## 👥 Contributing

This is an academic research project for thesis submission. Contributions are welcome after initial publication.

**For collaborators:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-agent`)
3. Commit changes (`git commit -am 'Add new agent architecture'`)
4. Push to branch (`git push origin feature/new-agent`)
5. Create Pull Request

**Areas for contribution**:
- Additional baseline models (Transformer-based, GAN-augmented, etc.)
- Alternative agent architectures (CNN for spatial patterns, GNN for patient networks)
- Interpretability tools (SHAP, LIME, attention visualization)
- Deployment tools (ONNX export, edge device optimization)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data Usage**:
- MIMIC-IV: Requires PhysioNet credentialing and signed DUA
- PhysioNet CinC 2019: Requires PhysioNet account (freely available)

**Citation**:
If you use this code in your research, please cite:

```bibtex
@mastersthesis{antigravity2026,
  title={Antigravity: Multi-Agent Deep Learning for Early Sepsis Prediction},
  author={[Your Name]},
  year={2026},
  school={University of Technology Sydney},
  note={Master's Thesis}
}
```

---

## 📞 Contact

**Author**: [Your Name]
**Email**: [your.email@student.uts.edu.au]
**Institution**: University of Technology Sydney
**Supervisor**: [Supervisor Name]

**Project Timeline**: January 2026 - April 2026
**Thesis Defense**: May 2026 (Expected)

---

## 🙏 Acknowledgments

- **MIT Laboratory for Computational Physiology** for MIMIC-IV database access
- **PhysioNet** for hosting the Computing in Cardiology Challenge 2019
- **UTS Faculty of Engineering and IT** for computational resources
- **Supervisor** [Name] for guidance and support
- **OpenSep** contributors for SOFA calculation reference implementation

---

**Last Updated**: January 26, 2026
**Version**: 1.0.0 (Initial Release)

---

*This project is part of academic research. For questions about methodology or collaboration, please contact the author.*
