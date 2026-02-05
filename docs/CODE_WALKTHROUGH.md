# Code Walkthrough - Understanding What We Built

**Purpose:** Help you explain the code to your supervisor
**Tip:** Read this before any code review meeting

---

## Quick Overview

We built **3 main components**:

```
1. Data Pipeline     → Turns raw hospital data into ML-ready format
2. Multi-Agent Model → Three specialist neural networks + coordinator
3. Training Pipeline → Trains and evaluates the model
```

---

## File Structure Explained

```
Sepsis/
├── config/
│   └── data_config.yaml      ← "Settings file - what variables to use"
├── src/
│   ├── data/
│   │   ├── harmonization.py  ← "Translates MIMIC format to our format"
│   │   ├── sofa_calculator.py← "Calculates organ failure scores"
│   │   └── labeling.py       ← "Labels patients: sepsis or no sepsis"
│   └── models/
│       └── multi_agent.py    ← "The AI brain - 3 specialists + coordinator"
└── notebooks/
    ├── MIMIC_IV_Preprocessing_Batched.ipynb  ← "Processes patient data"
    ├── Train_MultiAgent_Model.ipynb          ← "Trains the AI"
    └── Baseline_Comparison.ipynb             ← "Compares to simpler methods"
```

---

## Component 1: Data Pipeline

### What it does:
> "Takes messy hospital data and converts it into clean, organized data that AI can learn from."

### File: `harmonization.py`

**In simple terms:**
> "MIMIC database uses item codes like '220045' for heart rate. This file translates those codes into readable names like 'heart_rate'."

**Key function to know:**
```python
def harmonize_patient(self, subject_id, chartevents, labevents, icu_intime, icu_outtime):
    # 1. Filter data to ICU stay period
    # 2. Map item codes to variable names (220045 → heart_rate)
    # 3. Resample to hourly intervals (one row per hour)
    # 4. Return clean DataFrame with columns like: hr, temp, sbp, creatinine...
```

**If asked "why hourly?":**
> "ICU data comes at irregular intervals - some vitals every 5 minutes, some labs every 6 hours. Resampling to hourly creates consistent time steps for the AI to learn from."

---

### File: `labeling.py`

**In simple terms:**
> "Determines which patients have sepsis using the official Sepsis-3 definition."

**The Sepsis-3 Definition (memorize this):**
```
Sepsis = Suspected Infection + Organ Failure (SOFA increase ≥ 2)
```

**Key function:**
```python
def label_patient(self, subject_id, data, prescriptions, microbiology, icu_intime, icu_outtime):
    # 1. Find suspected infection time (antibiotics + cultures)
    # 2. Calculate SOFA score over time
    # 3. Check if SOFA increased by 2+ near infection time
    # 4. If yes → sepsis_label = 1, else → sepsis_label = 0
```

**If asked "how do you know it's infection?":**
> "We look for the combination of antibiotics prescribed AND cultures ordered within 24 hours. This is the clinical definition of 'suspected infection'."

---

### File: `sofa_calculator.py`

**In simple terms:**
> "SOFA score measures how badly organs are failing. Higher score = sicker patient."

**SOFA components (6 organ systems):**
```
1. Respiration   → PaO2/FiO2 ratio
2. Coagulation   → Platelet count
3. Liver         → Bilirubin level
4. Cardiovascular→ Blood pressure, vasopressors
5. Neurological  → Glasgow Coma Scale
6. Renal         → Creatinine, urine output
```

**If asked "why SOFA?":**
> "It's the standard measure of organ dysfunction. The Sepsis-3 definition specifically uses SOFA increase ≥ 2 as the criterion for sepsis."

---

## Component 2: Multi-Agent Model

### File: `multi_agent.py`

**The Big Picture:**
```
Patient Data → [Vitals Agent]  →
             → [Labs Agent]    → [Meta-Learner] → Prediction (0-1)
             → [Trend Agent]   →
```

**If asked "why three agents?":**
> "Different data types have different characteristics:
> - Vitals are frequent and mostly complete
> - Labs are sparse with lots of missing values
> - Trends capture how values change over time
>
> Each agent is specialized for its data type."

---

### Agent 1: VitalsAgent

```python
class VitalsAgent(nn.Module):
    """Processes: hr, resp, temp, sbp, dbp, map_value, o2sat"""

    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2):
        # LSTM: Good at learning sequences (how vitals change over time)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        # Attention: Learns which time points matter most
        self.attention = nn.Linear(hidden_dim, 1)
```

**If asked "what's LSTM?":**
> "Long Short-Term Memory - a type of neural network designed for sequences. It remembers important information from earlier time points while processing new data. Perfect for time-series like vital signs."

**If asked "what's attention?":**
> "A mechanism that learns which time points are most important. For example, the hour before sepsis onset might be weighted higher than 20 hours before."

---

### Agent 2: LabsAgent

```python
class LabsAgent(nn.Module):
    """Processes: creatinine, lactate, bilirubin, etc. (17 features)"""

    def __init__(self, input_dim=17, hidden_dim=64):
        # Learned imputation: Handles missing values intelligently
        self.imputation_layer = nn.Linear(input_dim, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim)
```

**If asked "how do you handle missing labs?":**
> "Labs have 60-99% missing values. Instead of just filling with averages, we have a 'learned imputation' layer that learns the best way to fill missing values based on the available data."

---

### Agent 3: TrendAgent

```python
class TrendAgent(nn.Module):
    """Processes: All 24 features - looks for temporal patterns"""

    def __init__(self, input_dim=24, hidden_dim=64):
        # Transformer: State-of-the-art for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

**If asked "what's a Transformer?":**
> "The same architecture behind ChatGPT. It's excellent at finding patterns across long sequences. It can directly compare any two time points, not just consecutive ones."

**If asked "why use Transformer here but LSTM for vitals?":**
> "Transformers are more powerful but need more data. For the trend agent looking at all 24 features together, the extra power helps. For individual agents with fewer features, LSTM is sufficient and more efficient."

---

### Meta-Learner (Coordinator)

```python
class MetaLearner(nn.Module):
    """Combines the three agents' outputs"""

    def forward(self, vitals_out, labs_out, trend_out):
        # Stack the three agent outputs
        agent_outputs = torch.stack([vitals_out, labs_out, trend_out])

        # Attention: Learn which agent to trust more
        weights = F.softmax(self.attention(agent_outputs), dim=0)

        # Weighted combination
        combined = (weights * agent_outputs).sum(dim=0)

        # Final prediction
        return self.classifier(combined)
```

**If asked "how does it combine the agents?":**
> "It uses attention to learn weights for each agent. In our results, each agent gets about 33% weight - meaning all three contribute equally. If one agent was useless, its weight would drop to near zero."

---

### Focal Loss

```python
class FocalLoss(nn.Module):
    """Handles class imbalance - sepsis is ~33% of data"""

    def forward(self, inputs, targets):
        # Standard cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Focal term: Down-weight easy examples
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
```

**If asked "why focal loss?":**
> "With ~33% sepsis cases, the model might just predict 'no sepsis' all the time and still be 67% accurate. Focal loss forces the model to focus on harder cases it gets wrong, especially the minority class (sepsis)."

---

## Component 3: Training Pipeline

### Notebook: `Train_MultiAgent_Model.ipynb`

**The Training Loop (simplified):**
```python
for epoch in range(50):
    # Training phase
    for batch in train_loader:
        predictions = model(batch)           # Forward pass
        loss = focal_loss(predictions, labels)  # Calculate error
        loss.backward()                      # Backpropagation
        optimizer.step()                     # Update weights

    # Validation phase
    val_auroc = evaluate(model, val_loader)

    # Early stopping
    if val_auroc > best_auroc:
        save_model()
        best_auroc = val_auroc
    elif no_improvement_for_10_epochs:
        break  # Stop training
```

**If asked "what's backpropagation?":**
> "The algorithm that calculates how much each weight contributed to the error, then adjusts weights to reduce the error. It's how neural networks learn."

**If asked "what's early stopping?":**
> "We monitor performance on validation data (patients not used for training). If performance stops improving for 10 epochs, we stop training to prevent overfitting."

---

## Key Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `sequence_length=24` | 24 hours of history | Captures daily patterns |
| `hidden_dim=64` | Network width | Balance between capacity and overfitting |
| `num_layers=2` | Network depth | Enough to learn complex patterns |
| `dropout=0.3` | 30% dropout | Regularization - prevents overfitting |
| `learning_rate=1e-4` | 0.0001 | Found through experimentation |
| `batch_size=64` | 64 samples per update | Standard for this data size |

**If asked "how did you choose these?":**
> "We started with standard values from literature, then ran experiments. For example, we tried learning rates of 1e-3, 1e-4, and 1e-5 - and found 1e-4 worked best."

---

## Common Questions & Answers

### "Walk me through how a prediction is made"

> 1. Take 24 hours of patient data (vitals + labs)
> 2. VitalsAgent processes the 7 vital signs → outputs a 64-dim vector
> 3. LabsAgent processes the 17 lab values → outputs a 64-dim vector
> 4. TrendAgent processes all 24 features → outputs a 64-dim vector
> 5. MetaLearner combines the three vectors using attention weights
> 6. Final classifier outputs probability (0-1) of sepsis in next 6 hours

### "What's the most important part of the code?"

> "The multi-agent architecture in `multi_agent.py`. The key insight is that different data types need different processing - vitals are dense, labs are sparse, and trends need long-range pattern detection."

### "What would you do differently?"

> "I'd experiment with:
> - More data (we used 3,559 patients; MIMIC has 50,000+)
> - Pre-training on related tasks
> - Adding more feature engineering (rolling averages, rate of change)"

### "How do you prevent overfitting?"

> "Three ways:
> 1. **Dropout** - randomly zeros 30% of neurons during training
> 2. **Early stopping** - stop when validation performance plateaus
> 3. **Patient-level splits** - ensure no patient appears in both train and test"

---

## Quick Reference Card

**When explaining the project, remember:**

```
WHAT:   AI to predict sepsis 6 hours early
HOW:    3 specialist networks + coordinator
DATA:   MIMIC-IV, 3,559 patients, 420K observations
RESULT: AUROC 0.7263 (beats clinical scores)
```

**Key terms to know:**
- **AUROC** - Performance metric (0.5 = random, 1.0 = perfect)
- **SOFA** - Organ failure score used in sepsis definition
- **LSTM** - Neural network for sequences
- **Transformer** - Advanced sequence model (like ChatGPT)
- **Attention** - Mechanism to weight important features
- **Focal Loss** - Loss function for imbalanced classes

---

*Review this before any code explanation meeting!*
