# Training Configurations for All Experimental Versions

This document shows the exact configuration used for each experimental version (v1-v6).

To use these configurations in `Train_MultiAgent_Model.ipynb`, copy the desired CONFIG dictionary into the notebook's cell 7.

---

## v1: Baseline (725 patients)

```python
CONFIG = {
    # Data
    'data_file': f"{DATA_PATH}/mimic_processed_medium.h5",  # 725 patients
    'sequence_length': 24,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,

    # Model
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.3,

    # Training
    'batch_size': 32,
    'learning_rate': 1e-3,  # v1: 1e-3
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,

    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # Feature groups
    'vitals_features': ['hr', 'resp', 'temp', 'sbp', 'dbp', 'map_value', 'o2sat'],
    'labs_features': ['bun', 'chloride', 'creatinine', 'wbc', 'bicarbonate', 'platelets',
                      'magnesium', 'calcium', 'potassium', 'sodium', 'glucose',
                      'fio2', 'ph', 'paco2', 'pao2', 'lactate', 'bilirubin'],
}
```

**Result:** AUROC 0.7391

---

## v2: Scale Up Data (3,559 patients) - FAILED

```python
CONFIG = {
    # Data
    'data_file': f"{DATA_PATH}/mimic_processed_large.h5",  # 3,559 patients (5x increase)
    'sequence_length': 24,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,

    # Model
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.3,

    # Training
    'batch_size': 32,
    'learning_rate': 1e-3,  # v2: KEPT AT 1e-3 (too high for more data)
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,

    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # Feature groups
    'vitals_features': ['hr', 'resp', 'temp', 'sbp', 'dbp', 'map_value', 'o2sat'],
    'labs_features': ['bun', 'chloride', 'creatinine', 'wbc', 'bicarbonate', 'platelets',
                      'magnesium', 'calcium', 'potassium', 'sodium', 'glucose',
                      'fio2', 'ph', 'paco2', 'pao2', 'lactate', 'bilirubin'],
}
```

**Result:** AUROC 0.6743 (dropped 0.0648 from v1)
**Issue:** Learning rate too high for larger dataset

---

## v3: Tuned Learning Rate (3,559 patients) - WINNER ✓

```python
CONFIG = {
    # Data
    'data_file': f"{DATA_PATH}/mimic_processed_large.h5",  # 3,559 patients
    'sequence_length': 24,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,

    # Model
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.3,

    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,  # v3: LOWERED TO 1e-4 (key change!)
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,

    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # Feature groups
    'vitals_features': ['hr', 'resp', 'temp', 'sbp', 'dbp', 'map_value', 'o2sat'],
    'labs_features': ['bun', 'chloride', 'creatinine', 'wbc', 'bicarbonate', 'platelets',
                      'magnesium', 'calcium', 'potassium', 'sodium', 'glucose',
                      'fio2', 'ph', 'paco2', 'pao2', 'lactate', 'bilirubin'],
}
```

**Result:** AUROC 0.7263, AUPRC 0.6536
**This is the final model used for all results**

---

## v4: Class Weights Adjustment (3,559 patients)

```python
CONFIG = {
    # Data
    'data_file': f"{DATA_PATH}/mimic_processed_large.h5",
    'sequence_length': 24,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,

    # Model
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.3,

    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,

    # Loss
    'focal_alpha': 0.35,  # v4: INCREASED FROM 0.25 (penalize missed sepsis more)
    'focal_gamma': 2.0,

    # Feature groups
    'vitals_features': ['hr', 'resp', 'temp', 'sbp', 'dbp', 'map_value', 'o2sat'],
    'labs_features': ['bun', 'chloride', 'creatinine', 'wbc', 'bicarbonate', 'platelets',
                      'magnesium', 'calcium', 'potassium', 'sodium', 'glucose',
                      'fio2', 'ph', 'paco2', 'pao2', 'lactate', 'bilirubin'],
}
```

**Result:** AUROC 0.6912
**Conclusion:** Higher alpha made model too conservative, hurt overall performance

---

## v5: Higher Dropout (3,559 patients)

```python
CONFIG = {
    # Data
    'data_file': f"{DATA_PATH}/mimic_processed_large.h5",
    'sequence_length': 24,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,

    # Model
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.4,  # v5: INCREASED FROM 0.3 (reduce overfitting)

    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,

    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # Feature groups
    'vitals_features': ['hr', 'resp', 'temp', 'sbp', 'dbp', 'map_value', 'o2sat'],
    'labs_features': ['bun', 'chloride', 'creatinine', 'wbc', 'bicarbonate', 'platelets',
                      'magnesium', 'calcium', 'potassium', 'sodium', 'glucose',
                      'fio2', 'ph', 'paco2', 'pao2', 'lactate', 'bilirubin'],
}
```

**Result:** AUROC 0.7198
**Conclusion:** Higher dropout reduced capacity too much, slightly worse than v3

---

## v6: Simpler Model (3,559 patients)

```python
CONFIG = {
    # Data
    'data_file': f"{DATA_PATH}/mimic_processed_large.h5",
    'sequence_length': 24,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,

    # Model
    'hidden_dim': 32,  # v6: REDUCED FROM 64
    'num_layers': 1,   # v6: REDUCED FROM 2
    'dropout': 0.3,

    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,

    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # Feature groups
    'vitals_features': ['hr', 'resp', 'temp', 'sbp', 'dbp', 'map_value', 'o2sat'],
    'labs_features': ['bun', 'chloride', 'creatinine', 'wbc', 'bicarbonate', 'platelets',
                      'magnesium', 'calcium', 'potassium', 'sodium', 'glucose',
                      'fio2', 'ph', 'paco2', 'pao2', 'lactate', 'bilirubin'],
}
```

**Result:** AUROC 0.7204
**Conclusion:** Model too simple, not enough capacity for complex sepsis patterns

---

## Quick Reference Table

| Version | Data File | Patients | LR | Dropout | Hidden | Layers | Focal α | AUROC |
|---------|-----------|----------|-----|---------|--------|--------|---------|-------|
| v1 | medium.h5 | 725 | 1e-3 | 0.3 | 64 | 2 | 0.25 | 0.7391 |
| v2 | large.h5 | 3,559 | 1e-3 | 0.3 | 64 | 2 | 0.25 | 0.6743 |
| **v3** | **large.h5** | **3,559** | **1e-4** | **0.3** | **64** | **2** | **0.25** | **0.7263** |
| v4 | large.h5 | 3,559 | 1e-4 | 0.3 | 64 | 2 | 0.35 | 0.6912 |
| v5 | large.h5 | 3,559 | 1e-4 | 0.4 | 64 | 2 | 0.25 | 0.7198 |
| v6 | large.h5 | 3,559 | 1e-4 | 0.3 | 32 | 1 | 0.25 | 0.7204 |

---

## How to Use

1. Open `Train_MultiAgent_Model.ipynb` in Google Colab
2. Navigate to the configuration cell (cell 7)
3. Replace the CONFIG dictionary with the desired version from above
4. Run all cells to train the model

## Key Findings

1. **Learning rate is critical when scaling data**
   - v1 (725 patients, lr=1e-3): Good performance
   - v2 (3,559 patients, lr=1e-3): Performance dropped
   - v3 (3,559 patients, lr=1e-4): Performance recovered

2. **Focal loss defaults work well**
   - Increasing alpha to 0.35 hurt performance (v4)

3. **Model architecture is appropriately sized**
   - Higher dropout (v5) and simpler model (v6) both slightly worse than v3

4. **Winner: v3**
   - Learning rate 1e-4 is optimal for 3,559 patient dataset
   - All other hyperparameters at default values
