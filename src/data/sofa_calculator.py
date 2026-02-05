"""
SOFA Score Calculator for Sepsis-3 Definition

This module implements the Sequential Organ Failure Assessment (SOFA) score
calculation based on the Sepsis-3 definition (Singer et al., JAMA 2016).

The SOFA score evaluates dysfunction across 6 organ systems:
1. Respiratory (PaO2/FiO2 ratio, ventilation)
2. Coagulation (Platelets)
3. Liver (Bilirubin)
4. Cardiovascular (MAP, vasopressors)
5. CNS (Glasgow Coma Scale)
6. Renal (Creatinine, urine output)

Each component is scored 0-4, for a total range of 0-24 points.

References:
    Vincent JL, et al. "The SOFA (Sepsis-related Organ Failure Assessment)
    score to describe organ dysfunction/failure." Intensive Care Med. 1996.

    Singer M, et al. "The Third International Consensus Definitions for
    Sepsis and Septic Shock (Sepsis-3)." JAMA. 2016.

Author: [Your Name]
Date: January 2026
"""

from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import logging

# Setup logging
logger = logging.getLogger(__name__)


class SOFACalculator:
    """
    Calculate SOFA scores from clinical data.

    This calculator handles missing data and provides validation of input values
    against physiological ranges.

    Attributes:
        scoring_thresholds (Dict): Thresholds for each SOFA component
        baseline_strategy (str): How to calculate baseline SOFA

    Example:
        >>> calculator = SOFACalculator()
        >>> patient_data = pd.Series({
        ...     'pao2': 90,
        ...     'fio2': 0.4,
        ...     'platelets': 120,
        ...     'bilirubin': 2.5,
        ...     # ... other values
        ... })
        >>> total_sofa = calculator.calculate_total_sofa(patient_data)
        >>> print(f"Total SOFA Score: {total_sofa}")
    """

    def __init__(self, baseline_strategy: str = "minimum_first_24h"):
        """
        Initialize SOFA calculator.

        Args:
            baseline_strategy: Method for calculating baseline SOFA
                - "minimum_first_24h": Minimum score in first 24 hours
                - "admission": Score at ICU admission
                - "fixed_zero": Assume baseline of 0 (conservative)
        """
        self.baseline_strategy = baseline_strategy
        self.scoring_thresholds = self._load_thresholds()
        logger.info(f"SOFACalculator initialized with baseline strategy: {baseline_strategy}")

    def _load_thresholds(self) -> Dict:
        """
        Load scoring thresholds for each SOFA component.

        Returns:
            Dictionary containing threshold values for scoring
        """
        return {
            'respiratory': {
                'pao2_fio2_ranges': [(400, 0), (300, 1), (200, 2), (100, 3), (0, 4)],
                'ventilation_bonus': 2  # Add 2 if ventilated and PaO2/FiO2 < 200
            },
            'coagulation': {
                'platelet_ranges': [(150, 0), (100, 1), (50, 2), (20, 3), (0, 4)]
            },
            'liver': {
                'bilirubin_ranges': [(1.2, 0), (1.9, 1), (5.9, 2), (11.9, 3), (float('inf'), 4)]
            },
            'cardiovascular': {
                'map_threshold': 70,
                'vasopressor_ranges': {
                    'none': 0,
                    'dopamine_low': 2,  # ≤5 or any dobutamine
                    'dopamine_mid': 3,   # >5 or epi/norepi ≤0.1
                    'dopamine_high': 4   # >15 or epi/norepi >0.1
                }
            },
            'cns': {
                'gcs_ranges': [(15, 0), (14, 1), (12, 2), (9, 3), (5, 4), (0, 4)]
            },
            'renal': {
                'creatinine_ranges': [(1.2, 0), (1.9, 1), (3.4, 2), (4.9, 3), (float('inf'), 4)],
                'urine_output_thresholds': {500: 3, 200: 4}
            }
        }

    def calculate_respiratory_score(self,
                                    pao2: Optional[float],
                                    fio2: Optional[float],
                                    is_ventilated: bool = False) -> int:
        """
        Calculate respiratory SOFA component.

        Score based on PaO2/FiO2 ratio (P/F ratio):
        - ≥400: 0 points
        - <400: 1 point
        - <300: 2 points
        - <200 with mechanical ventilation: 3 points
        - <100 with mechanical ventilation: 4 points

        Args:
            pao2: Partial pressure of arterial oxygen (mmHg)
            fio2: Fraction of inspired oxygen (0-1 or 0-100)
            is_ventilated: Whether patient is mechanically ventilated

        Returns:
            Respiratory SOFA score (0-4)

        Notes:
            - If pao2 or fio2 is missing, returns 0 (conservative)
            - If fio2 > 1, assumes percentage and divides by 100
            - Mechanical ventilation status affects scoring when P/F < 200
        """
        # Handle missing values
        if pd.isna(pao2) or pd.isna(fio2):
            logger.warning("Missing pao2 or fio2, returning score 0")
            return 0

        # Normalize FiO2 to fraction (0-1)
        if fio2 > 1:
            fio2 = fio2 / 100.0

        # Calculate P/F ratio
        pf_ratio = pao2 / fio2

        # Score based on PaO2/FiO2 ratio thresholds
        if pf_ratio >= 400:
            score = 0
        elif pf_ratio >= 300:
            score = 1
        elif pf_ratio >= 200:
            score = 2
        elif pf_ratio >= 100:
            score = 3 if is_ventilated else 2
        else:  # pf_ratio < 100
            score = 4 if is_ventilated else 3

        return score

    def calculate_coagulation_score(self, platelets: Optional[float]) -> int:
        """
        Calculate coagulation SOFA component.

        Score based on platelet count (×10³/μL):
        - ≥150: 0 points
        - <150: 1 point
        - <100: 2 points
        - <50: 3 points
        - <20: 4 points

        Args:
            platelets: Platelet count (×10³/μL)

        Returns:
            Coagulation SOFA score (0-4)
        """
        if pd.isna(platelets):
            return 0

        if platelets >= 150:
            return 0
        elif platelets >= 100:
            return 1
        elif platelets >= 50:
            return 2
        elif platelets >= 20:
            return 3
        else:
            return 4

    def calculate_liver_score(self, bilirubin: Optional[float]) -> int:
        """
        Calculate liver SOFA component.

        Score based on total bilirubin (mg/dL):
        - <1.2: 0 points
        - 1.2-1.9: 1 point
        - 2.0-5.9: 2 points
        - 6.0-11.9: 3 points
        - ≥12.0: 4 points

        Args:
            bilirubin: Total bilirubin (mg/dL)

        Returns:
            Liver SOFA score (0-4)
        """
        if pd.isna(bilirubin):
            return 0

        if bilirubin < 1.2:
            return 0
        elif bilirubin < 2.0:
            return 1
        elif bilirubin < 6.0:
            return 2
        elif bilirubin < 12.0:
            return 3
        else:
            return 4

    def calculate_cardiovascular_score(self,
                                       map_value: Optional[float],
                                       dopamine: float = 0.0,
                                       dobutamine: float = 0.0,
                                       epinephrine: float = 0.0,
                                       norepinephrine: float = 0.0) -> int:
        """
        Calculate cardiovascular SOFA component.

        Score based on MAP and vasopressor requirements:
        - MAP ≥70: 0 points
        - MAP <70: 1 point
        - Dopamine ≤5 or any dobutamine: 2 points
        - Dopamine >5 OR epi ≤0.1 OR norepi ≤0.1: 3 points
        - Dopamine >15 OR epi >0.1 OR norepi >0.1: 4 points

        Args:
            map_value: Mean arterial pressure (mmHg)
            dopamine: Dopamine dose (μg/kg/min)
            dobutamine: Dobutamine dose (μg/kg/min)
            epinephrine: Epinephrine dose (μg/kg/min)
            norepinephrine: Norepinephrine dose (μg/kg/min)

        Returns:
            Cardiovascular SOFA score (0-4)

        Notes:
            Vasopressor doses are in μg/kg/min
        """
        if pd.isna(map_value):
            map_value = 0  # Conservative: assume hypotensive if missing

        # Check for high-dose vasopressors (score 4)
        if dopamine > 15 or epinephrine > 0.1 or norepinephrine > 0.1:
            return 4

        # Check for medium-dose vasopressors (score 3)
        if dopamine > 5 or (epinephrine > 0 and epinephrine <= 0.1) or \
           (norepinephrine > 0 and norepinephrine <= 0.1):
            return 3

        # Check for low-dose vasopressors (score 2)
        if (dopamine > 0 and dopamine <= 5) or dobutamine > 0:
            return 2

        # No vasopressors, score based on MAP
        if map_value >= 70:
            return 0
        else:
            return 1

    def calculate_cns_score(self, gcs: Optional[float]) -> int:
        """
        Calculate CNS SOFA component based on Glasgow Coma Scale.

        Score based on GCS:
        - 15: 0 points
        - 13-14: 1 point
        - 10-12: 2 points
        - 6-9: 3 points
        - <6: 4 points

        Args:
            gcs: Glasgow Coma Scale total score (3-15)

        Returns:
            CNS SOFA score (0-4)

        Notes:
            - GCS is sum of Eye (1-4) + Verbal (1-5) + Motor (1-6)
            - If total GCS not available, can calculate from components
        """
        if pd.isna(gcs):
            logger.warning("Missing GCS, returning score 0")
            return 0

        if gcs == 15:
            return 0
        elif gcs >= 13:
            return 1
        elif gcs >= 10:
            return 2
        elif gcs >= 6:
            return 3
        else:
            return 4

    def calculate_renal_score(self,
                             creatinine: Optional[float],
                             urine_output: Optional[float] = None) -> int:
        """
        Calculate renal SOFA component.

        Score based on creatinine (mg/dL) and/or urine output (mL/day):
        - Creatinine <1.2: 0 points
        - Creatinine 1.2-1.9: 1 point
        - Creatinine 2.0-3.4: 2 points
        - Creatinine 3.5-4.9 OR UO <500: 3 points
        - Creatinine ≥5.0 OR UO <200: 4 points

        Args:
            creatinine: Serum creatinine (mg/dL)
            urine_output: Urine output (mL/day), optional

        Returns:
            Renal SOFA score (0-4)

        Notes:
            - If both available, uses the higher score
            - Urine output typically calculated as 24-hour total
        """
        score_creat = 0
        score_uo = 0

        if not pd.isna(creatinine):
            if creatinine < 1.2:
                score_creat = 0
            elif creatinine < 2.0:
                score_creat = 1
            elif creatinine < 3.5:
                score_creat = 2
            elif creatinine < 5.0:
                score_creat = 3
            else:
                score_creat = 4

        if not pd.isna(urine_output):
            if urine_output < 200:
                score_uo = 4
            elif urine_output < 500:
                score_uo = 3
            else:
                score_uo = 0

        return max(score_creat, score_uo)

    def calculate_total_sofa(self, patient_data: pd.Series) -> int:
        """
        Calculate total SOFA score from all components.

        Args:
            patient_data: Series containing required clinical variables:
                - pao2, fio2, is_ventilated (respiratory)
                - platelets (coagulation)
                - bilirubin (liver)
                - map_value, vasopressors (cardiovascular)
                - gcs (CNS)
                - creatinine, urine_output (renal)

        Returns:
            Total SOFA score (0-24)

        Example:
            >>> patient = pd.Series({
            ...     'pao2': 85, 'fio2': 0.5, 'is_ventilated': True,
            ...     'platelets': 95, 'bilirubin': 3.2, 'map_value': 65,
            ...     'gcs': 12, 'creatinine': 2.8
            ... })
            >>> total = calculator.calculate_total_sofa(patient)
        """
        scores = {
            'respiratory': self.calculate_respiratory_score(
                patient_data.get('pao2'),
                patient_data.get('fio2'),
                patient_data.get('is_ventilated', False)
            ),
            'coagulation': self.calculate_coagulation_score(
                patient_data.get('platelets')
            ),
            'liver': self.calculate_liver_score(
                patient_data.get('bilirubin')
            ),
            'cardiovascular': self.calculate_cardiovascular_score(
                patient_data.get('map_value'),
                patient_data.get('dopamine', 0.0),
                patient_data.get('dobutamine', 0.0),
                patient_data.get('epinephrine', 0.0),
                patient_data.get('norepinephrine', 0.0)
            ),
            'cns': self.calculate_cns_score(
                patient_data.get('gcs')
            ),
            'renal': self.calculate_renal_score(
                patient_data.get('creatinine'),
                patient_data.get('urine_output')
            )
        }

        total_score = sum(scores.values())

        logger.debug(f"SOFA component scores: {scores}")
        logger.info(f"Total SOFA score: {total_score}")

        return total_score

    def calculate_baseline_sofa(self,
                                patient_df: pd.DataFrame,
                                icu_intime: pd.Timestamp) -> int:
        """
        Calculate baseline SOFA score for a patient.

        Baseline is typically defined as the minimum SOFA score in the first
        24 hours of ICU admission, representing the patient's "normal" state.

        Args:
            patient_df: DataFrame with patient's time-series data
            icu_intime: ICU admission timestamp

        Returns:
            Baseline SOFA score (0-24)

        Notes:
            - Uses strategy specified in __init__ (default: minimum_first_24h)
            - Critical for identifying SOFA increase ≥2 (organ dysfunction)
        """
        if self.baseline_strategy == "minimum_first_24h":
            # Calculate SOFA for first 24 hours
            first_24h = patient_df[
                (patient_df['charttime'] >= icu_intime) &
                (patient_df['charttime'] < icu_intime + pd.Timedelta(hours=24))
            ]

            if len(first_24h) == 0:
                logger.warning("No data in first 24h, returning baseline SOFA = 0")
                return 0

            # Calculate SOFA for each timepoint
            sofa_scores = []
            for idx, row in first_24h.iterrows():
                sofa = self.calculate_total_sofa(row)
                sofa_scores.append(sofa)

            # Return minimum SOFA in first 24h
            baseline = min(sofa_scores) if sofa_scores else 0
            logger.info(f"Baseline SOFA (min first 24h): {baseline}")
            return baseline

        elif self.baseline_strategy == "admission":
            # Use SOFA at admission time
            admission_data = patient_df[
                patient_df['charttime'] == icu_intime
            ]
            if len(admission_data) == 0:
                # If no data at exact admission, use first available
                admission_data = patient_df.iloc[0:1]

            return self.calculate_total_sofa(admission_data.iloc[0])

        elif self.baseline_strategy == "fixed_zero":
            return 0

        else:
            raise ValueError(f"Unknown baseline strategy: {self.baseline_strategy}")

    def calculate_delta_sofa(self,
                            current_sofa: int,
                            baseline_sofa: int) -> int:
        """
        Calculate change in SOFA score from baseline.

        Critical for Sepsis-3 definition:
        - Delta SOFA ≥2 indicates organ dysfunction
        - Organ dysfunction + suspected infection = Sepsis

        Args:
            current_sofa: SOFA score at current time
            baseline_sofa: Baseline SOFA score

        Returns:
            Delta SOFA score (can be negative if improvement)

        Example:
            >>> baseline = 2
            >>> current = 5
            >>> delta = calculator.calculate_delta_sofa(current, baseline)
            >>> print(delta)  # 3
            >>> print(f"Organ dysfunction: {delta >= 2}")  # True
        """
        return current_sofa - baseline_sofa

    def detect_organ_dysfunction(self,
                                delta_sofa: int,
                                threshold: int = 2) -> bool:
        """
        Determine if delta SOFA indicates organ dysfunction.

        Sepsis-3 definition requires SOFA increase ≥2 from baseline.

        Args:
            delta_sofa: Change in SOFA score from baseline
            threshold: Threshold for organ dysfunction (default: 2)

        Returns:
            True if organ dysfunction detected, False otherwise
        """
        return delta_sofa >= threshold


# Validation functions
def validate_sofa_inputs(patient_data: pd.Series) -> Dict[str, bool]:
    """
    Validate that SOFA input values are within physiological ranges.

    Args:
        patient_data: Series with clinical measurements

    Returns:
        Dictionary of validation results for each variable
    """
    validations = {}

    # Define physiological ranges
    ranges = {
        'pao2': (0, 700),
        'fio2': (0.21, 1.0),
        'platelets': (0, 1000),
        'bilirubin': (0, 50),
        'map_value': (0, 250),
        'gcs': (3, 15),
        'creatinine': (0, 30),
        'urine_output': (0, 10000)
    }

    for var, (min_val, max_val) in ranges.items():
        if var in patient_data and not pd.isna(patient_data[var]):
            value = patient_data[var]
            validations[var] = (min_val <= value <= max_val)

    return validations


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    calculator = SOFACalculator()

    # Example patient data
    example_patient = pd.Series({
        'pao2': 95,
        'fio2': 0.4,
        'is_ventilated': False,
        'platelets': 120,
        'bilirubin': 2.3,
        'map_value': 68,
        'gcs': 13,
        'creatinine': 1.8
    })

    # Calculate total SOFA
    total_sofa = calculator.calculate_total_sofa(example_patient)
    print(f"\nExample Total SOFA Score: {total_sofa}")

    # Validate inputs
    validations = validate_sofa_inputs(example_patient)
    print(f"\nValidation Results: {validations}")
