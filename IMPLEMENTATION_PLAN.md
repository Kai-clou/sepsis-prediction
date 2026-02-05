# Implementation Plan: Antigravity Multi-Agent Sepsis Prediction System

## Project Context

**Objective:** Create a multi-agent deep learning system to predict sepsis 6-12 hours before clinical recognition, comparing performance against single-model baselines.

**Deliverable:** Academic thesis with working code demonstrating that multi-agent architecture (temporal LSTM + static FFN + time-to-event regression) outperforms traditional single-model approaches.

**Critical Deadline:** Show tangible progress to supervisor by **February 5, 2026** (~10 days)

**Current State:** Empty project directory with only `Sepsis.md`

**Data Status:** MIMIC-IV and PhysioNet CinC 2019 available on Google Drive

---

## Immediate Deliverables for Supervisor Meeting (By Feb 5, 2026)

To demonstrate substantial progress, we will create a **complete structural workflow** with the following tangible files:

### Phase 1A: Core Documentation (3 files)
1. **README.md** (750 lines) - Professional project documentation
2. **IMPLEMENTATION_PLAN.md** (1200 lines) - Detailed technical specification
3. **docs/data_dictionary.md** (200 lines) - MIMIC-IV ↔ CinC variable mappings

### Phase 1B: Configuration Files (5 files)
4. **requirements.txt** (30 lines) - Python dependencies with versions
5. **.gitignore** (40 lines) - Exclude data/models/experiments
6. **config/data_config.yaml** (150 lines) - Harmonization settings, itemid mappings
7. **config/model_config.yaml** (80 lines) - Multi-agent architecture hyperparameters
8. **config/training_config.yaml** (70 lines) - Training settings, MLflow config

### Phase 1C: Project Structure (42 folders)
Create complete directory tree with all subfolders:
- `src/` with 5 submodules (data, models, training, evaluation, utils)
- `data/` with 3 subdirectories (raw, processed, metadata)
- `scripts/` for 8 executable workflows
- `notebooks/` for 4 exploratory analyses
- `tests/`, `docs/`, `results/` directories

### Phase 1D: Skeleton Python Files (10 critical modules with docstrings)
9. **src/data/sofa_calculator.py** (300 lines) - SOFA score class with method signatures
10. **src/data/harmonization.py** (250 lines) - MIMIC→CinC mapping class
11. **src/data/labeling.py** (200 lines) - Sepsis-3 definition implementation
12. **src/models/agent_a.py** (150 lines) - Temporal LSTM agent class
13. **src/models/agent_b.py** (100 lines) - Static FFN agent class
14. **src/models/agent_c.py** (80 lines) - Time-to-event regression head
15. **src/models/multi_agent_system.py** (200 lines) - Complete architecture
16. **src/models/baselines.py** (150 lines) - Single LSTM, XGBoost baselines
17. **src/training/trainer.py** (300 lines) - Training loop with MLflow
18. **src/training/loss_functions.py** (100 lines) - Multi-task loss

### Phase 1E: Workflow Scripts (8 executable scripts)
19. **scripts/01_download_data.py** - Google Drive download automation
20. **scripts/02_preprocess_mimic.py** - MIMIC-IV harmonization pipeline
21. **scripts/03_preprocess_cinc.py** - CinC 2019 processing
22. **scripts/04_train_multi_agent.py** - Multi-agent training workflow
23. **scripts/05_train_baselines.py** - Baseline model training
24. **scripts/06_evaluate_internal.py** - MIMIC-IV test evaluation
25. **scripts/07_evaluate_external.py** - CinC 2019 external validation
26. **scripts/08_generate_thesis_plots.py** - Figure generation

### Phase 1F: Exploratory Notebooks (4 Jupyter notebooks)
27. **notebooks/01_eda_mimic.ipynb** - MIMIC-IV data exploration
28. **notebooks/02_eda_cinc.ipynb** - CinC 2019 data exploration
29. **notebooks/03_sofa_validation.ipynb** - SOFA calculation validation
30. **notebooks/04_model_comparison.ipynb** - Results comparison

**Total Deliverables: 30 files + 42 folders = Complete structural workflow**

### What Your Supervisor Will See

Opening the project folder, they will find:
```
antigravity/
├── README.md ✅ (Polished documentation)
├── IMPLEMENTATION_PLAN.md ✅ (This detailed plan)
├── requirements.txt ✅
├── .gitignore ✅
├── config/ ✅ (3 YAML files with real hyperparameters)
├── data/ ✅ (Organized subdirectories)
├── src/ ✅ (5 submodules with 10 skeleton Python files)
├── scripts/ ✅ (8 executable workflows)
├── notebooks/ ✅ (4 Jupyter notebooks)
├── docs/ ✅ (data_dictionary.md)
├── tests/ ✅
└── results/ ✅
```

### What Makes This "Structural Workflow"

1. **Not just ideas** - Actual files they can open and read
2. **Architecture is defined** - Classes, methods, parameters all specified
3. **Clear execution path** - scripts/01→02→03→04 shows the workflow
4. **Ready to code** - Skeleton files have TODOs where implementation goes
5. **Professional presentation** - Looks like a mature research project

This demonstrates you've moved from "just an idea" to a **well-architected, implementation-ready project**.

---

## Phase 1: Documentation (Primary Focus)

### 1.1 README.md
Create a comprehensive GitHub-style README with:

- **Project Overview:** "Antigravity" multi-agent sepsis prediction system
- **Key Innovation:** Compare multi-agent (Agent A: LSTM, Agent B: FFN, Agent C: TTE) vs single-model baselines
- **Datasets:** MIMIC-IV (training) + PhysioNet CinC 2019 (external validation)
- **Architecture Diagram:** ASCII or described architecture showing agent fusion
- **Quick Start Guide:** Installation, data download, training commands
- **Project Structure:** Annotated directory tree
- **Methodology Summary:** Sepsis-3 definition, SOFA calculation, prediction window (6-12h)
- **Expected Results:** Target metrics (AUROC 0.82-0.88 internal, 0.75-0.82 external)
- **References:** Key papers (Sepsis-3 definition, OpenSep, CinC 2019 challenge)

### 1.2 IMPLEMENTATION_PLAN.md
Detailed technical specification covering:

- **ETL Pipeline Design:**
  - Data loading from Google Drive
  - MIMIC-IV → CinC schema harmonization (itemid mapping)
  - Unit conversions (Temperature F→C, Lactate mmol/L vs mg/dL)
  - SOFA score calculation (6 components)
  - Sepsis-3 labeling (infection suspicion + organ dysfunction)
  - Missing data imputation (forward fill + mean fallback)
  - Temporal alignment (irregular → hourly bins)

- **Model Architecture:**
  - Agent A: Bi-directional LSTM (input: [batch, 24h, 40 features])
  - Agent B: Feed-forward network (input: [batch, demographics])
  - Fusion layer: Concatenate + FC layers
  - Classification head: Binary sepsis prediction
  - Agent C: Time-to-event regression (masked MSE loss)

- **Training Pipeline:**
  - Multi-task loss: weighted BCE + masked MSE
  - Class imbalance handling (pos_weight=10-15)
  - MLflow experiment tracking
  - Hyperparameter configs (LSTM hidden=128, layers=2, dropout=0.3)

- **Evaluation Framework:**
  - Internal validation: MIMIC-IV held-out test set
  - External validation: CinC 2019 full dataset
  - Metrics: AUROC, AUPRC, Sensitivity@90% Specificity, PhysioNet Utility Score
  - Baseline comparisons: Single LSTM, XGBoost, Logistic Regression

### 1.3 docs/data_dictionary.md
Variable mapping reference:

| CinC Variable | MIMIC-IV itemid | Source Table | Unit | Notes |
|---------------|-----------------|--------------|------|-------|
| HR | 220045 | chartevents | bpm | Heart Rate |
| SBP | 220179, 220050 | chartevents | mmHg | Non-invasive + Arterial |
| Temp | 223762 (C), 223761 (F) | chartevents | Celsius | Requires F→C conversion |
| O2Sat | 220277 | chartevents | % | SpO2 |
| Lactate | 50813 | labevents | mmol/L | Arterial Lactate |
| Creatinine | 50912 | labevents | mg/dL | Renal SOFA component |
| Bilirubin | 50885 | labevents | mg/dL | Liver SOFA component |
| Platelets | 51265 | labevents | ×10³/μL | Coagulation SOFA component |

(Include all 40 CinC variables with mappings)

---

## Phase 2: Project Structure Setup

### 2.1 Directory Tree
Create the following structure:

```
antigravity/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── evaluation_config.yaml
├── data/
│   ├── raw/
│   │   ├── mimic_iv/
│   │   └── cinc2019/
│   ├── processed/
│   │   ├── mimic_harmonized/
│   │   └── cinc_processed/
│   └── metadata/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mimic_loader.py
│   │   ├── cinc_loader.py
│   │   ├── harmonization.py
│   │   ├── sofa_calculator.py
│   │   ├── labeling.py
│   │   ├── imputation.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── agent_a.py
│   │   ├── agent_b.py
│   │   ├── agent_c.py
│   │   ├── fusion.py
│   │   ├── multi_agent_system.py
│   │   └── baselines.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── loss_functions.py
│   │   ├── metrics.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── internal_validation.py
│   │   ├── external_validation.py
│   │   └── utility_score.py
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py
│       └── logging_utils.py
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_preprocess_mimic.py
│   ├── 03_preprocess_cinc.py
│   ├── 04_train_multi_agent.py
│   ├── 05_train_baselines.py
│   ├── 06_evaluate_internal.py
│   ├── 07_evaluate_external.py
│   └── 08_generate_thesis_plots.py
├── notebooks/
│   ├── 01_eda_mimic.ipynb
│   ├── 02_eda_cinc.ipynb
│   ├── 03_sofa_validation.ipynb
│   └── 04_model_comparison.ipynb
├── tests/
│   ├── test_harmonization.py
│   ├── test_sofa_calculation.py
│   └── test_models.py
├── docs/
│   ├── methodology.md
│   ├── data_dictionary.md
│   └── reproducibility_guide.md
└── results/
    ├── figures/
    ├── checkpoints/
    └── predictions/
```

### 2.2 Configuration Files

**requirements.txt:**
```
# Core ML/DL
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Data Processing
h5py>=3.8.0
PyYAML>=6.0

# Clinical Data
wfdb>=4.1.0  # PhysioNet data formats

# Experiment Tracking
mlflow>=2.9.0
optuna>=3.5.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0

# Utilities
tqdm>=4.65.0
python-dotenv>=1.0.0
```

**.gitignore:**
```
# Data (too large for git)
data/raw/
data/processed/
*.h5
*.hdf5
*.csv
*.csv.gz
*.psv

# Model artifacts
results/checkpoints/
results/predictions/
mlruns/
*.pth
*.pkl

# Python
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/

# Environment
.env
venv/
env/

# IDE
.vscode/
.idea/
*.swp
```

**config/model_config.yaml:**
```yaml
# Multi-Agent Architecture Configuration

agent_a_temporal:
  type: "BiLSTM"
  input_size: 40  # 40 CinC canonical variables
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true
  use_attention: true  # Optional attention mechanism

agent_b_static:
  type: "FFN"
  input_size: 15  # Demographics (age, sex, admission type, etc.)
  hidden_layers: [64, 32]
  dropout: 0.2
  activation: "relu"

fusion:
  hidden_size: 128
  dropout: 0.3

classification_head:
  hidden_layers: [64]
  dropout: 0.15
  activation: "relu"

agent_c_tte:
  hidden_layers: [64]
  dropout: 0.2
  output_activation: "sigmoid"  # Normalize to [0,1]

# Baseline Configurations
baseline_lstm:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3

baseline_xgboost:
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 100
  subsample: 0.8
```

**config/data_config.yaml:**
```yaml
# Data Sources
mimic_iv:
  path: "data/raw/mimic_iv/"
  version: "2.2"
  required_tables:
    - chartevents
    - labevents
    - prescriptions
    - microbiologyevents
    - patients
    - admissions
    - icustays

cinc_2019:
  path: "data/raw/cinc2019/"
  training_sets: ["training_setA", "training_setB"]
  test_set: "test"

# Harmonization Settings
variable_mapping:
  # Vitals
  HR: [220045]
  Resp: [220210, 224690]
  Temp: [223761, 223762]  # F and C
  SBP: [220179, 220050]
  MAP: [220052, 220181]
  O2Sat: [220277]

  # Labs
  Lactate: [50813]
  Creatinine: [50912]
  Bilirubin: [50885]
  Platelets: [51265]
  WBC: [51301]

unit_conversions:
  temperature:
    fahrenheit_to_celsius: true
    itemids: [223761]

  lactate:
    mg_dl_to_mmol_l: true
    conversion_factor: 0.111

# Temporal Settings
temporal_alignment:
  bin_size_hours: 1
  lookback_window_hours: 24
  aggregation:
    vitals: "median"  # Robust to outliers
    labs: "last"      # Most recent value

# Labeling
sepsis_definition:
  type: "sepsis3"
  prediction_window:
    early_hours: 12
    optimal_hours: 6

  infection_suspicion:
    antibiotic_culture_window_hours: 24

  sofa:
    baseline_calculation: "minimum_first_24h"
    delta_threshold: 2

# Missing Data
imputation:
  strategy: "hybrid"  # forward_fill + mean fallback
  forward_fill_max_hours:
    vitals: 6
    labs: 24
  add_missingness_indicators: true

# Data Splits
splits:
  train: 0.70
  val: 0.15
  test: 0.15
  stratify_by: "sepsis_label"
  random_seed: 42
```

**config/training_config.yaml:**
```yaml
# Training Configuration

training:
  max_epochs: 100
  batch_size: 256
  num_workers: 4
  pin_memory: true

optimizer:
  type: "AdamW"
  learning_rate: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]

scheduler:
  type: "ReduceLROnPlateau"
  mode: "max"
  factor: 0.5
  patience: 5
  min_lr: 0.00001

loss:
  classification_weight: 0.7
  tte_weight: 0.3
  pos_class_weight: 12.0  # Adjust based on actual class imbalance
  focal_loss: false  # Alternative to weighted BCE

regularization:
  gradient_clip_norm: 1.0
  dropout: 0.3
  weight_decay: 0.0001

early_stopping:
  patience: 10
  monitor: "val_auroc"
  mode: "max"

checkpointing:
  save_best: true
  save_frequency: 5  # epochs

mlflow:
  experiment_name: "antigravity_multi_agent"
  tracking_uri: "./mlruns"
  log_models: true

reproducibility:
  random_seed: 42
  deterministic: true
  benchmark: false
```

---

## Phase 3: Skeleton Python Files (Key Modules)

Create template files with docstrings and function signatures for the most critical modules:

### 3.1 src/data/sofa_calculator.py
```python
"""
SOFA Score Calculator for Sepsis-3 Definition

Implements Sequential Organ Failure Assessment (SOFA) score calculation
based on Singer et al., JAMA 2016 (Sepsis-3 definition).

SOFA Components:
1. Respiratory (PaO2/FiO2 ratio, ventilation)
2. Coagulation (Platelets)
3. Liver (Bilirubin)
4. Cardiovascular (MAP, vasopressors)
5. CNS (Glasgow Coma Scale)
6. Renal (Creatinine, urine output)

Score Range: 0-24 (0-4 per component)
"""

from typing import Optional, Dict
import pandas as pd
import numpy as np


class SOFACalculator:
    """Calculate SOFA scores from clinical data."""

    def __init__(self):
        """Initialize SOFA calculator with scoring thresholds."""
        self.scoring_thresholds = self._load_thresholds()

    def calculate_respiratory_score(self,
                                    pao2: float,
                                    fio2: float,
                                    is_ventilated: bool) -> int:
        """
        Calculate respiratory SOFA component.

        Args:
            pao2: Partial pressure of oxygen (mmHg)
            fio2: Fraction of inspired oxygen (0-1 or 0-100)
            is_ventilated: Whether patient is mechanically ventilated

        Returns:
            Respiratory SOFA score (0-4)
        """
        pass

    def calculate_coagulation_score(self, platelets: float) -> int:
        """
        Calculate coagulation SOFA component.

        Args:
            platelets: Platelet count (×10³/μL)

        Returns:
            Coagulation SOFA score (0-4)
        """
        pass

    # [Additional methods for other SOFA components...]

    def calculate_total_sofa(self, patient_data: pd.Series) -> int:
        """
        Calculate total SOFA score from all components.

        Args:
            patient_data: Series containing all required clinical variables

        Returns:
            Total SOFA score (0-24)
        """
        pass

    def calculate_delta_sofa(self,
                            current_sofa: int,
                            baseline_sofa: int) -> int:
        """
        Calculate change in SOFA score from baseline.

        Critical for Sepsis-3 definition (delta ≥2 indicates organ dysfunction).

        Args:
            current_sofa: SOFA score at current time
            baseline_sofa: Baseline SOFA (minimum in first 24h ICU)

        Returns:
            Delta SOFA score
        """
        return current_sofa - baseline_sofa
```

### 3.2 src/data/harmonization.py
```python
"""
Data Harmonization: MIMIC-IV → PhysioNet CinC 2019 Schema

Maps MIMIC-IV's complex schema (itemids, irregular timestamps) to
CinC 2019's flat hourly schema (40 canonical variables).

Key Functions:
- Variable mapping (itemid → CinC variable name)
- Unit conversion (Temperature F→C, Lactate mg/dL→mmol/L)
- Temporal alignment (irregular → hourly bins)
"""

import pandas as pd
from typing import Dict, List
import yaml


class MIMICHarmonizer:
    """Harmonize MIMIC-IV data to CinC 2019 schema."""

    def __init__(self, config_path: str):
        """
        Initialize harmonizer with variable mappings.

        Args:
            config_path: Path to data_config.yaml
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.variable_mapping = self.config['variable_mapping']
        self.unit_conversions = self.config['unit_conversions']

    def map_itemids_to_variables(self,
                                 chartevents: pd.DataFrame,
                                 labevents: pd.DataFrame) -> pd.DataFrame:
        """
        Map MIMIC-IV itemids to CinC canonical variables.

        Example: itemid 220045 → HR (Heart Rate)

        Args:
            chartevents: MIMIC-IV chartevents table
            labevents: MIMIC-IV labevents table

        Returns:
            DataFrame with CinC variable names as columns
        """
        pass

    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply unit conversions to ensure consistency.

        Conversions:
        - Temperature: Fahrenheit → Celsius
        - Lactate: mg/dL → mmol/L (if needed)

        Args:
            df: DataFrame with mixed units

        Returns:
            DataFrame with standardized units
        """
        pass

    def create_hourly_bins(self,
                          df: pd.DataFrame,
                          patient_id: str,
                          icu_intime: pd.Timestamp) -> pd.DataFrame:
        """
        Align irregular timestamps to hourly bins.

        MIMIC-IV has observations at irregular intervals (every few minutes).
        CinC 2019 expects hourly observations.

        Strategy:
        - Create 1-hour bins from ICU admission
        - Aggregate vitals: median (robust to outliers)
        - Aggregate labs: last (most recent value)

        Args:
            df: Patient data with irregular timestamps
            patient_id: Patient identifier
            icu_intime: ICU admission timestamp

        Returns:
            DataFrame with hourly observations
        """
        pass
```

### 3.3 src/models/multi_agent_system.py
```python
"""
Antigravity Multi-Agent Sepsis Prediction System

Architecture:
- Agent A: Bidirectional LSTM for temporal features (vitals/labs)
- Agent B: Feed-forward network for static features (demographics)
- Fusion Layer: Combines Agent A and B representations
- Classification Head: Binary sepsis prediction
- Agent C: Time-to-event regression (auxiliary task)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .agent_a import TemporalLSTMAgent
from .agent_b import StaticFFNAgent
from .agent_c import TimeToEventAgent


class AntigravityMultiAgent(nn.Module):
    """Complete multi-agent architecture for sepsis prediction."""

    def __init__(self, config: dict):
        """
        Initialize multi-agent system.

        Args:
            config: Model configuration dictionary (from model_config.yaml)
        """
        super().__init__()

        # Initialize agents
        self.agent_a = TemporalLSTMAgent(
            input_size=config['agent_a_temporal']['input_size'],
            hidden_size=config['agent_a_temporal']['hidden_size'],
            num_layers=config['agent_a_temporal']['num_layers'],
            dropout=config['agent_a_temporal']['dropout'],
            bidirectional=config['agent_a_temporal']['bidirectional']
        )

        self.agent_b = StaticFFNAgent(
            input_size=config['agent_b_static']['input_size'],
            hidden_layers=config['agent_b_static']['hidden_layers'],
            dropout=config['agent_b_static']['dropout']
        )

        # Fusion layer
        fusion_input_size = (
            self.agent_a.output_size + self.agent_b.output_size
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, config['fusion']['hidden_size']),
            nn.BatchNorm1d(config['fusion']['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config['fusion']['dropout'])
        )

        # Classification head
        self.classifier = self._build_classifier(config['classification_head'])

        # Agent C (time-to-event)
        self.agent_c = TimeToEventAgent(
            input_size=config['fusion']['hidden_size'],
            config=config['agent_c_tte']
        )

    def forward(self,
                temporal_x: torch.Tensor,
                static_x: torch.Tensor,
                temporal_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-agent system.

        Args:
            temporal_x: [batch, seq_len, temporal_features] - Time-series data
            static_x: [batch, static_features] - Demographics
            temporal_mask: [batch, seq_len] - Padding mask for sequences

        Returns:
            sepsis_prob: [batch, 1] - Predicted sepsis probability
            time_to_event: [batch, 1] - Predicted hours until onset
        """
        # Agent encodings
        temporal_features = self.agent_a(temporal_x, temporal_mask)
        static_features = self.agent_b(static_x)

        # Fuse representations
        fused = torch.cat([temporal_features, static_features], dim=1)
        fused = self.fusion(fused)

        # Predictions
        sepsis_prob = self.classifier(fused)
        time_to_event = self.agent_c(fused)

        return sepsis_prob, time_to_event

    def _build_classifier(self, config: dict) -> nn.Module:
        """Build classification head from config."""
        pass
```

### 3.4 src/training/trainer.py
```python
"""
Training Orchestration with MLflow Integration

Handles:
- Training loop with multi-task loss
- Validation and early stopping
- MLflow experiment tracking
- Model checkpointing
"""

import torch
import mlflow
from typing import Dict, Tuple
from torch.utils.data import DataLoader


class MultiAgentTrainer:
    """Trainer for Antigravity multi-agent system."""

    def __init__(self,
                 model: torch.nn.Module,
                 config: Dict,
                 experiment_name: str = "antigravity"):
        """
        Initialize trainer.

        Args:
            model: AntigravityMultiAgent instance
            config: Training configuration (from training_config.yaml)
            experiment_name: MLflow experiment name
        """
        self.model = model
        self.config = config

        # Setup optimizer
        self.optimizer = self._build_optimizer()

        # Setup loss function
        self.loss_fn = self._build_loss_function()

        # MLflow tracking
        mlflow.set_experiment(experiment_name)
        self.mlflow_run = None

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int) -> Dict[str, float]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs

        Returns:
            Dictionary of final metrics
        """
        with mlflow.start_run() as run:
            self.mlflow_run = run

            # Log configuration
            mlflow.log_params(self.config)

            best_val_auroc = 0.0
            patience_counter = 0

            for epoch in range(epochs):
                # Training
                train_metrics = self._train_epoch(train_loader, epoch)

                # Validation
                val_metrics = self._validate(val_loader, epoch)

                # Log metrics
                mlflow.log_metrics(train_metrics, step=epoch)
                mlflow.log_metrics(val_metrics, step=epoch)

                # Early stopping check
                if val_metrics['auroc'] > best_val_auroc:
                    best_val_auroc = val_metrics['auroc']
                    patience_counter = 0
                    self._save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1

                if patience_counter >= self.config['early_stopping']['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break

            return val_metrics

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Single training epoch."""
        self.model.train()
        # Implementation...
        pass

    def _validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validation pass."""
        self.model.eval()
        # Implementation...
        pass
```

---

## Phase 4: Critical Files to Implement First (Post-Supervisor Meeting)

After the February 5 meeting, prioritize implementing these modules in order:

### Week 1 (Feb 5-12): Data Pipeline
1. **src/data/mimic_loader.py** - Load data from Google Drive
2. **src/data/harmonization.py** - Variable mapping and unit conversion
3. **src/data/sofa_calculator.py** - SOFA score implementation
4. **src/data/labeling.py** - Sepsis-3 ground truth generation

**Milestone:** `data/processed/` contains harmonized HDF5 files

### Week 2 (Feb 12-19): Model Development
1. **src/models/agent_a.py** - Temporal LSTM agent
2. **src/models/agent_b.py** - Static FFN agent
3. **src/models/multi_agent_system.py** - Complete architecture
4. **src/models/baselines.py** - Single LSTM baseline

**Milestone:** Can instantiate model and run forward pass

### Week 3 (Feb 19-26): Training Infrastructure
1. **src/training/loss_functions.py** - Multi-task loss
2. **src/training/trainer.py** - Training loop with MLflow
3. **scripts/04_train_multi_agent.py** - Main training script
4. **scripts/05_train_baselines.py** - Baseline training

**Milestone:** Training runs converge, MLflow logs metrics

### Week 4 (Feb 26-Mar 5): Evaluation
1. **src/evaluation/internal_validation.py** - MIMIC-IV test evaluation
2. **src/evaluation/utility_score.py** - PhysioNet utility metric
3. **scripts/06_evaluate_internal.py** - Internal validation script
4. **scripts/07_evaluate_external.py** - CinC 2019 validation

**Milestone:** Complete results table comparing all models

---

## Key Technical Decisions

### Data Harmonization Strategy
**Decision:** Force MIMIC-IV to match CinC schema (not vice versa)
**Rationale:** CinC is the external validation target; harmonizing MIMIC ensures direct comparability

### Handling Class Imbalance
**Decision:** Use weighted BCE loss (pos_weight=12) + oversampling
**Rationale:** Sepsis prevalence ~4-7%; standard BCE would predict all negative

### Missing Data Imputation
**Decision:** Forward fill (up to 6h for vitals, 24h for labs) + population mean fallback
**Rationale:** Clinical standard practice; captures "last known state" assumption

### SOFA Baseline Calculation
**Decision:** Minimum SOFA in first 24h of ICU admission
**Rationale:** Aligned with Sepsis-3 definition; identifies "normal" baseline for patient

### Prediction Window (6-12h)
**Decision:** Label time steps 6-12h before sepsis onset as positive
**Rationale:** Clinically actionable window; earlier predictions have too many false alarms

### Multi-Task Learning Weights
**Decision:** α=0.7 (classification), β=0.3 (time-to-event)
**Rationale:** Primary goal is classification; TTE provides auxiliary supervision

### Train/Val/Test Split
**Decision:** 70/15/15 stratified by sepsis outcome
**Rationale:** Maintains class balance across splits; sufficient validation set for early stopping

---

## Verification Plan

After implementation, verify system works end-to-end:

### 1. Data Pipeline Verification
```bash
python scripts/02_preprocess_mimic.py
# Expected output: HDF5 files in data/processed/mimic_harmonized/
# Verify: 40 variables, hourly timestamps, SOFA scores calculated
```

### 2. SOFA Calculation Validation
```python
# Compare against MIMIC-IV official sepsis table
# notebooks/03_sofa_validation.ipynb
# Expected: >95% agreement with published sepsis cohorts
```

### 3. Model Training Verification
```bash
python scripts/04_train_multi_agent.py
# Expected: Training loss decreases, val AUROC increases
# Check MLflow UI: mlflow ui --backend-store-uri ./mlruns
```

### 4. Baseline Comparison Verification
```bash
python scripts/05_train_baselines.py
# Expected: Multi-agent AUROC > Single LSTM > XGBoost > Logistic
```

### 5. External Validation Verification
```bash
python scripts/07_evaluate_external.py
# Expected: CinC AUROC within 5-10% of MIMIC AUROC (domain shift)
```

---

## Expected Timeline to Completion

| Phase | Duration | End Date | Deliverable |
|-------|----------|----------|-------------|
| **Phase 1:** Project Setup | 1 week | Feb 5 | Documentation + structure (THIS PHASE) |
| **Phase 2:** Data Pipeline | 2 weeks | Feb 19 | Harmonized datasets |
| **Phase 3:** Model Development | 2 weeks | Mar 5 | Trained models |
| **Phase 4:** Evaluation | 1 week | Mar 12 | Results tables |
| **Phase 5:** Thesis Writing | 3 weeks | Apr 2 | Draft thesis |
| **Phase 6:** Revision | 2 weeks | Apr 16 | Final submission |

**Total:** ~11 weeks (~3 months) with 2-week buffer before deadline

---

## Risk Mitigation

### Risk 1: SOFA Calculation Discrepancies
**Mitigation:** Validate against MIMIC-IV's pre-computed sepsis table
**Fallback:** Use MIMIC's official sepsis labels (less novel but safer)

### Risk 2: Poor Multi-Agent Performance
**Mitigation:** Start with strong baselines (validate XGBoost reproduces literature)
**Fallback:** Pivot to "comprehensive benchmark study" if multi-agent underperforms

### Risk 3: Data Download Issues
**Mitigation:** Test Google Drive download early (Week 1)
**Fallback:** Use MIMIC-Extract (preprocessed subset) if full download fails

### Risk 4: Computational Resources
**Mitigation:** Request university GPU cluster access
**Fallback:** Google Colab Pro ($10/month) or AWS SageMaker (~$100 budget)

---

## Critical Files List

**Immediate Creation (for Feb 5 meeting):**
1. README.md - Project overview and documentation
2. IMPLEMENTATION_PLAN.md - This document
3. requirements.txt - Python dependencies
4. .gitignore - Exclude data/models
5. config/data_config.yaml - Data harmonization settings
6. config/model_config.yaml - Architecture hyperparameters
7. config/training_config.yaml - Training settings
8. docs/data_dictionary.md - Variable mapping reference
9. Complete directory structure (all folders)
10. Skeleton files for 4 critical modules (sofa_calculator, harmonization, multi_agent_system, trainer)

**Post-Meeting Implementation Priority:**
1. src/data/sofa_calculator.py
2. src/data/harmonization.py
3. src/models/multi_agent_system.py
4. src/training/trainer.py

---

## Success Criteria for Supervisor Meeting

✅ **Demonstrates Research:** References to Sepsis-3, MIMIC-IV, CinC 2019, multi-agent learning
✅ **Shows Architecture Design:** Clear separation of agents (temporal/static/TTE)
✅ **Plans Data Pipeline:** Detailed harmonization strategy with itemid mappings
✅ **Professional Setup:** Clean directory structure, configs, documentation
✅ **Addresses Challenges:** Unit conversion, missing data, class imbalance
✅ **Sets Milestones:** Clear weekly plan from now until submission
✅ **Includes Evaluation:** Internal + external validation, multiple baselines

This demonstrates you've moved from "just an idea" to a **well-researched, architected project** ready for implementation.