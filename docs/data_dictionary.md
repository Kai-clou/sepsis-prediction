# Data Dictionary: MIMIC-IV ↔ PhysioNet CinC 2019 Variable Mapping

**Document Version**: 1.0.0
**Last Updated**: January 26, 2026
**Purpose**: Complete reference for harmonizing MIMIC-IV database to PhysioNet/Computing in Cardiology Challenge 2019 schema

---

## Table of Contents

1. [Overview](#overview)
2. [Vital Signs](#vital-signs)
3. [Laboratory Values](#laboratory-values)
4. [Arterial Blood Gas](#arterial-blood-gas)
5. [SOFA Components](#sofa-components)
6. [Demographics](#demographics)
7. [Unit Conversions](#unit-conversions)
8. [Missing Data Codes](#missing-data-codes)
9. [Data Quality Notes](#data-quality-notes)

---

## Overview

This document maps **MIMIC-IV itemids** to the **40 canonical variables** required by the PhysioNet/Computing in Cardiology Challenge 2019. The CinC 2019 dataset provides a flat, hourly schema with consistent variable names and units across multiple hospital systems.

### Key Principles

1. **Schema Target**: MIMIC-IV is transformed to match CinC 2019 (not vice versa)
2. **Multiple itemids**: A single CinC variable may correspond to multiple MIMIC itemids (e.g., invasive vs non-invasive blood pressure)
3. **Unit Consistency**: All values are converted to CinC-specified units
4. **Temporal Alignment**: MIMIC's irregular timestamps are binned into hourly observations

### MIMIC-IV Table Sources

- **chartevents**: Vital signs, ventilator settings (most frequent observations)
- **labevents**: Laboratory measurements (less frequent, typically every 6-24h)
- **inputevents_mv**: Medication infusions (e.g., vasopressors)
- **outputevents**: Urine output, drainage
- **patients**: Demographics (age, sex)
- **admissions**: Admission details
- **icustays**: ICU stay metadata

---

## Vital Signs

Vital signs are sourced from the **chartevents** table. These are typically measured every 1-5 minutes in ICU settings.

| CinC Variable | MIMIC-IV itemid(s) | Label in MIMIC | Unit | Normal Range | Notes |
|---------------|-------------------|----------------|------|--------------|-------|
| **HR** | 220045 | Heart Rate | beats/min | 60-100 | Most common HR itemid in MIMIC-IV |
| **O2Sat** | 220277, 220227 | Oxygen Saturation | % | 95-100 | SpO2 from pulse oximetry |
| **Temp** | 223762 (C), 223761 (F) | Temperature | °C | 36.5-37.5 | **Requires conversion** if Fahrenheit |
| **SBP** | 220179, 220050 | Systolic BP | mmHg | 90-120 | 220179=non-invasive, 220050=arterial |
| **DBP** | 220180, 220051 | Diastolic BP | mmHg | 60-80 | 220180=non-invasive, 220051=arterial |
| **MAP** | 220052, 220181, 225312 | Mean Arterial Pressure | mmHg | 70-100 | Critical for SOFA cardiovascular component |
| **Resp** | 220210, 224690 | Respiratory Rate | breaths/min | 12-20 | 224690="Respiratory Rate (Total)" |

### Aggregation Strategy for Vitals

When multiple measurements exist within a 1-hour bin:
- **Use median**: Robust to outliers and transient artifacts
- **Example**: If HR measurements in hour are [78, 82, 150 (artifact), 80], median = 80

### Blood Pressure Hierarchy

If multiple BP measurements are available, prioritize:
1. **Arterial line** (220050, 220051, 220052) - most accurate
2. **Non-invasive** (220179, 220180, 220181) - standard measurement
3. If both exist in same hour, prefer arterial line

---

## Laboratory Values

Laboratory values are sourced from the **labevents** table. These are measured intermittently (typically every 6-24 hours).

### Basic Metabolic Panel

| CinC Variable | MIMIC-IV itemid(s) | Label in MIMIC | Unit | Normal Range | Clinical Significance |
|---------------|-------------------|----------------|------|--------------|----------------------|
| **Creatinine** | 50912 | Creatinine | mg/dL | 0.7-1.3 | **Renal SOFA component** |
| **BUN** | 51006 | Blood Urea Nitrogen | mg/dL | 7-20 | Kidney function |
| **Glucose** | 50809, 50931 | Glucose | mg/dL | 70-110 | 50809=serum, 50931=whole blood |
| **Sodium** | 50824, 50983 | Sodium | mEq/L | 136-145 | Electrolyte balance |
| **Potassium** | 50822, 50971 | Potassium | mEq/L | 3.5-5.0 | Critical for cardiac function |
| **Chloride** | 50806, 50902 | Chloride | mEq/L | 96-106 | Acid-base balance |
| **Bicarbonate** | 50882 | Bicarbonate | mEq/L | 22-29 | Buffer system |

### Complete Blood Count (CBC)

| CinC Variable | MIMIC-IV itemid(s) | Label in MIMIC | Unit | Normal Range | Clinical Significance |
|---------------|-------------------|----------------|------|--------------|----------------------|
| **WBC** | 51300, 51301 | White Blood Cells | ×10³/μL | 4-11 | Infection marker |
| **Platelets** | 51265 | Platelet Count | ×10³/μL | 150-400 | **Coagulation SOFA component** |
| **Hematocrit** | 51221, 50810 | Hematocrit | % | 37-47 (F), 40-54 (M) | Oxygen-carrying capacity |
| **Hemoglobin** | 51222, 50811 | Hemoglobin | g/dL | 12-16 (F), 14-18 (M) | Oxygen-carrying capacity |

### Liver Function Tests

| CinC Variable | MIMIC-IV itemid(s) | Label in MIMIC | Unit | Normal Range | Clinical Significance |
|---------------|-------------------|----------------|------|--------------|----------------------|
| **Bilirubin_total** | 50885 | Bilirubin, Total | mg/dL | 0.1-1.2 | **Liver SOFA component** |
| **AST** | 50878 | Aspartate Aminotransferase | U/L | 10-40 | Liver enzyme |
| **ALT** | 50861 | Alanine Aminotransferase | U/L | 7-56 | Liver enzyme |
| **Alkaline_phosphatase** | 50863 | Alkaline Phosphatase | U/L | 45-115 | Liver/bone enzyme |

### Other Critical Labs

| CinC Variable | MIMIC-IV itemid(s) | Label in MIMIC | Unit | Normal Range | Clinical Significance |
|---------------|-------------------|----------------|------|--------------|----------------------|
| **Lactate** | 50813 | Lactate | mmol/L | 0.5-2.2 | **Tissue hypoxia marker**, critical for sepsis |
| **Magnesium** | 50960 | Magnesium | mg/dL | 1.7-2.2 | Electrolyte balance |
| **Calcium** | 50893 | Calcium, Total | mg/dL | 8.5-10.5 | Electrolyte balance |
| **Ionized_calcium** | 50808 | Calcium, Ionized | mmol/L | 1.15-1.35 | Active form of calcium |

### Aggregation Strategy for Labs

When multiple measurements exist within a 1-hour bin:
- **Use last (most recent)**: Labs change slowly; most recent value is most informative
- **Exception**: If values differ dramatically (>50%), investigate for data quality issues

---

## Arterial Blood Gas (ABG)

ABG measurements are critical for assessing respiratory function and acid-base status. Sourced from **labevents** table.

| CinC Variable | MIMIC-IV itemid(s) | Label in MIMIC | Unit | Normal Range | Clinical Significance |
|---------------|-------------------|----------------|------|--------------|----------------------|
| **pH** | 50820 | pH (Arterial) | pH units | 7.35-7.45 | Acid-base status |
| **PaO2** | 50821 | Oxygen, Partial Pressure (Arterial) | mmHg | 75-100 | **Respiratory SOFA component** |
| **PaCO2** | 50818 | Carbon Dioxide, Partial Pressure | mmHg | 35-45 | Ventilation status |
| **Base_excess** | 50803 | Base Excess | mEq/L | -2 to +2 | Metabolic acid-base |
| **FiO2** | 50816, 223835 | Fraction Inspired Oxygen | % or 0-1 | 21 (room air) | **Required for PaO2/FiO2 ratio** |

### FiO2 Normalization

- **If FiO2 > 1**: Divide by 100 (convert percentage to fraction)
- **If missing**: Assume FiO2=0.21 (room air) if no supplemental oxygen documented
- **Ventilated patients**: FiO2 typically 0.40-1.0

### PaO2/FiO2 Ratio Calculation

```
PaO2/FiO2 ratio = PaO2 (mmHg) / FiO2 (fraction)

Example:
  PaO2 = 90 mmHg
  FiO2 = 0.4 (40%)
  Ratio = 90 / 0.4 = 225 mmHg
```

**Clinical Interpretation**:
- >400: Normal respiratory function
- 200-300: Mild hypoxemia
- 100-200: Moderate hypoxemia (ARDS criteria)
- <100: Severe hypoxemia

---

## SOFA Components

The Sequential Organ Failure Assessment (SOFA) score requires specific variables from multiple sources.

### Respiratory SOFA

| Variable | Source | MIMIC itemid | Calculation |
|----------|--------|--------------|-------------|
| PaO2 | labevents | 50821 | From ABG |
| FiO2 | labevents, chartevents | 50816, 223835 | Supplemental oxygen |
| Mechanical Ventilation | chartevents | 225792, 225794 | Any ventilator mode entry |

**Score Calculation**:
- PaO2/FiO2 ≥ 400: **0 points**
- PaO2/FiO2 < 400: **1 point**
- PaO2/FiO2 < 300: **2 points**
- PaO2/FiO2 < 200 + ventilation: **3 points**
- PaO2/FiO2 < 100 + ventilation: **4 points**

### Coagulation SOFA

| Variable | Source | MIMIC itemid | Unit |
|----------|--------|--------------|------|
| Platelets | labevents | 51265 | ×10³/μL |

**Score Calculation**:
- Platelets ≥ 150: **0 points**
- Platelets < 150: **1 point**
- Platelets < 100: **2 points**
- Platelets < 50: **3 points**
- Platelets < 20: **4 points**

### Liver SOFA

| Variable | Source | MIMIC itemid | Unit |
|----------|--------|--------------|------|
| Bilirubin | labevents | 50885 | mg/dL |

**Score Calculation**:
- Bilirubin < 1.2: **0 points**
- Bilirubin 1.2-1.9: **1 point**
- Bilirubin 2.0-5.9: **2 points**
- Bilirubin 6.0-11.9: **3 points**
- Bilirubin ≥ 12.0: **4 points**

### Cardiovascular SOFA

| Variable | Source | MIMIC itemid | Unit |
|----------|--------|--------------|------|
| MAP | chartevents | 220052, 220181 | mmHg |
| Dopamine | inputevents_mv | 221662 | μg/kg/min |
| Dobutamine | inputevents_mv | 221653 | μg/kg/min |
| Epinephrine | inputevents_mv | 221289 | μg/kg/min |
| Norepinephrine | inputevents_mv | 221906 | μg/kg/min |

**Score Calculation**:
- MAP ≥ 70: **0 points**
- MAP < 70: **1 point**
- Dopamine ≤ 5 or dobutamine (any dose): **2 points**
- Dopamine > 5 OR epinephrine ≤ 0.1 OR norepinephrine ≤ 0.1: **3 points**
- Dopamine > 15 OR epinephrine > 0.1 OR norepinephrine > 0.1: **4 points**

### CNS SOFA

| Variable | Source | MIMIC itemid | Unit |
|----------|--------|--------------|------|
| GCS Total | chartevents | 220739 | 3-15 |
| GCS Eye | chartevents | 228578 | 1-4 |
| GCS Verbal | chartevents | 228576 | 1-5 |
| GCS Motor | chartevents | 228577 | 1-6 |

**Score Calculation**:
- GCS = 15: **0 points**
- GCS 13-14: **1 point**
- GCS 10-12: **2 points**
- GCS 6-9: **3 points**
- GCS < 6: **4 points**

**Note**: If total GCS not available, calculate as Eye + Verbal + Motor.

### Renal SOFA

| Variable | Source | MIMIC itemid | Unit |
|----------|--------|--------------|------|
| Creatinine | labevents | 50912 | mg/dL |
| Urine Output | outputevents | 226559-226567 | mL/day |

**Score Calculation**:
- Creatinine < 1.2: **0 points**
- Creatinine 1.2-1.9: **1 point**
- Creatinine 2.0-3.4: **2 points**
- Creatinine 3.5-4.9 OR urine output < 500 mL/day: **3 points**
- Creatinine ≥ 5.0 OR urine output < 200 mL/day: **4 points**

---

## Demographics

Demographic features are sourced from **patients** and **admissions** tables.

| CinC Variable | MIMIC-IV Source | Table | Notes |
|---------------|-----------------|-------|-------|
| **Age** | `anchor_age` | patients | Actual age = anchor_age + (admittime.year - anchor_year) |
| **Gender** | `gender` | patients | M/F (1/0 encoding) |
| **Admission_type** | `admission_type` | admissions | Emergency, Urgent, Elective |
| **ICU_type** | `first_careunit` | icustays | MICU, SICU, CCU, CSRU, etc. |
| **Insurance** | `insurance` | admissions | Medicare, Medicaid, Private |
| **Ethnicity** | `ethnicity` | admissions | Grouped into categories |

### Age De-identification

MIMIC-IV uses **anchor year** system for de-identification:
- Ages > 89 are masked to 91
- True age must be calculated using anchor_age and anchor_year

```python
# Age calculation
actual_age = anchor_age + (admission_year - anchor_year)

# Clip to realistic range
age = min(actual_age, 91)  # Cap at 91 for de-identification
```

### Categorical Encoding

**Gender**:
- Male = 1
- Female = 0

**Admission Type** (One-hot encoding):
- Emergency = [1, 0, 0]
- Urgent = [0, 1, 0]
- Elective = [0, 0, 1]

**ICU Type** (One-hot encoding):
- MICU (Medical) = [1, 0, 0, 0]
- SICU (Surgical) = [0, 1, 0, 0]
- CCU (Cardiac) = [0, 0, 1, 0]
- Other = [0, 0, 0, 1]

---

## Unit Conversions

### Temperature: Fahrenheit → Celsius

MIMIC-IV itemid **223761** records temperature in Fahrenheit. Must convert to Celsius for CinC schema.

```python
def fahrenheit_to_celsius(temp_f):
    return (temp_f - 32) * 5/9

# Example
temp_f = 98.6  # Normal body temp in F
temp_c = (98.6 - 32) * 5/9 = 37.0  # °C
```

**Validation**:
- If converted Celsius temp < 30 or > 45, flag as potential error
- Normal ICU range: 35-40°C

### Lactate: mg/dL → mmol/L (if needed)

**Note**: MIMIC-IV already stores lactate in **mmol/L**, so conversion is typically not needed. However, some external datasets use mg/dL.

```python
def mg_dl_to_mmol_l(lactate_mg_dl):
    return lactate_mg_dl * 0.111

# Example
lactate_mg_dl = 18  # mg/dL
lactate_mmol_l = 18 * 0.111 = 2.0  # mmol/L
```

### Glucose: mmol/L → mg/dL (if needed)

MIMIC-IV stores glucose in **mg/dL** (matching CinC). No conversion needed.

For reference:
```python
def mmol_l_to_mg_dl(glucose_mmol_l):
    return glucose_mmol_l * 18.0
```

---

## Missing Data Codes

MIMIC-IV does not use explicit missing data codes. Missing values are simply absent from the tables. However, some erroneous values should be treated as missing:

### Physiologically Implausible Values

Treat as missing if:

| Variable | Implausible Range | Action |
|----------|------------------|--------|
| HR | < 0 or > 300 | Set to NaN |
| Temp | < 30°C or > 45°C | Set to NaN |
| SBP | < 0 or > 300 | Set to NaN |
| DBP | < 0 or > 200 | Set to NaN |
| MAP | < 0 or > 250 | Set to NaN |
| Resp | < 0 or > 70 | Set to NaN |
| O2Sat | < 0 or > 100 | Set to NaN |
| Platelets | < 0 or > 1000 | Set to NaN |
| Lactate | < 0 or > 30 | Set to NaN |

### Error Values

Some MIMIC itemids have documented error codes:
- **-1**: Documented error code in some vital signs
- **0**: Sometimes indicates "not applicable" rather than true zero

**Recommended approach**: Treat values outside physiological ranges as missing.

---

## Data Quality Notes

### Temporal Resolution Differences

| Dataset | Vitals Frequency | Labs Frequency | Duration |
|---------|-----------------|----------------|----------|
| **MIMIC-IV** | 1-5 minutes | 6-24 hours | Variable (hours to weeks) |
| **CinC 2019** | 1 hour | 1 hour | Fixed (maximum 72 hours) |

**Harmonization Impact**:
- MIMIC's high-frequency vitals are downsampled to hourly (information loss)
- CinC expects hourly labs (MIMIC may have gaps > 1 hour)

### Missingness Patterns

Typical missingness in ICU data:

| Variable Type | Expected Missingness | Reason |
|--------------|---------------------|--------|
| Vitals (HR, BP, Temp) | 5-15% | Equipment disconnections, patient movement |
| Routine Labs (CBC, BMP) | 30-50% | Ordered daily or less frequently |
| Specialized Labs (Lactate) | 60-80% | Ordered only when clinically indicated |
| ABG | 70-90% | Invasive test, ordered for acute changes |
| Urine Output | 20-40% | May not be catheterized |

**Clinical Interpretation**: **Informative missingness** — labs are more likely missing when patient is stable.

### Duplicate Measurements

When multiple measurements exist at the same timestamp:
1. **Vitals**: Use median (robust to errors)
2. **Labs**: Prefer arterial over venous, prefer chemistry analyzer over point-of-care

### Data Entry Errors

Common patterns to watch for:
- **Decimal point errors**: Creatinine = 100 mg/dL (likely meant 1.00)
- **Unit confusion**: Glucose = 5 (likely mmol/L reported as mg/dL)
- **Copy-paste errors**: Identical values repeated across multiple hours

**Mitigation**: Apply physiological range checks and statistical outlier detection.

---

## References

### MIMIC-IV Documentation
- [MIMIC-IV Official Documentation](https://mimic.mit.edu/docs/iv/)
- [MIMIC-IV Data Tables](https://mimic.mit.edu/docs/iv/modules/)
- [MIMIC Code Repository](https://github.com/MIT-LCP/mimic-code)

### PhysioNet CinC 2019
- [Challenge Description](https://physionet.org/content/challenge-2019/1.0.0/)
- [Variable Definitions](https://physionet.org/content/challenge-2019/1.0.0/#files)

### Clinical References
- **SOFA Score**: Vincent JL, et al. *Intensive Care Med* 1996. DOI: 10.1007/BF01709751
- **Sepsis-3**: Singer M, et al. *JAMA* 2016. DOI: 10.1001/jama.2016.0287

---

**Document Maintenance**: This data dictionary should be updated whenever:
1. New MIMIC-IV itemids are discovered for existing variables
2. CinC challenge releases new variable definitions
3. Harmonization logic is modified
4. Data quality issues are identified

**Last Review Date**: January 26, 2026
**Next Scheduled Review**: February 26, 2026
