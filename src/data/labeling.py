"""
Sepsis-3 Labeling Implementation

This module implements the Sepsis-3 definition from Singer et al., JAMA 2016:
    Sepsis = Suspected Infection + Organ Dysfunction

Components:
1. Suspected Infection: Antibiotic + Culture within 24-hour window
2. Organ Dysfunction: SOFA score increase ≥2 from baseline

Prediction Window:
- Positive labels: 6-12 hours before sepsis onset
- Negative labels: No sepsis during ICU stay

Author: Jason
Date: January 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import timedelta

from .sofa_calculator import SOFACalculator

logger = logging.getLogger(__name__)


class SepsisLabeler:
    """Generate Sepsis-3 labels for MIMIC-IV ICU patients."""

    def __init__(self, config: Dict):
        """
        Initialize Sepsis labeler.

        Args:
            config: Configuration dict from data_config.yaml['sepsis_definition']
        """
        self.config = config
        self.sofa_calc = SOFACalculator(
            baseline_strategy=config['sofa']['baseline_calculation']
        )

        self.prediction_window_early = config['prediction_window']['early_hours']
        self.prediction_window_optimal = config['prediction_window']['optimal_hours']

        self.antibiotic_culture_window = config['infection_suspicion'][
            'antibiotic_culture_window_hours'
        ]

        logger.info(f"SepsisLabeler initialized with prediction window: "
                   f"{self.prediction_window_optimal}-{self.prediction_window_early}h before onset")

    def detect_suspected_infection(self,
                                   prescriptions: pd.DataFrame,
                                   microbiology: pd.DataFrame,
                                   icu_intime: pd.Timestamp,
                                   icu_outtime: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        Detect suspected infection using antibiotic + culture orders.

        Sepsis-3 definition: Suspected infection = antibiotics + culture
        within 24-hour window (either order can come first).

        Args:
            prescriptions: Patient's prescription data
            microbiology: Patient's microbiology culture data
            icu_intime: ICU admission time
            icu_outtime: ICU discharge time

        Returns:
            Timestamp of suspected infection onset, or None if no infection suspected
        """
        # Filter to ICU stay period
        prescriptions_icu = prescriptions[
            (prescriptions['starttime'] >= icu_intime) &
            (prescriptions['starttime'] <= icu_outtime)
        ]

        microbiology_icu = microbiology[
            (microbiology['charttime'] >= icu_intime) &
            (microbiology['charttime'] <= icu_outtime)
        ]

        # Filter to antibiotics
        antibiotic_keywords = ['antibiotic', 'antimicrobial', 'penicillin',
                              'cephalosporin', 'vancomycin', 'meropenem',
                              'piperacillin', 'ciprofloxacin', 'azithromycin']

        abx_mask = prescriptions_icu['drug_type'].str.lower().str.contains(
            '|'.join(antibiotic_keywords), na=False
        ) | prescriptions_icu['drug'].str.lower().str.contains(
            '|'.join(antibiotic_keywords), na=False
        )

        antibiotics = prescriptions_icu[abx_mask].copy()

        if len(antibiotics) == 0 or len(microbiology_icu) == 0:
            return None

        # Check all combinations for temporal proximity
        window_hours = self.antibiotic_culture_window

        for _, abx in antibiotics.iterrows():
            abx_time = abx['starttime']

            for _, culture in microbiology_icu.iterrows():
                culture_time = culture['charttime']

                # Check if within window (either direction)
                time_diff_hours = abs((culture_time - abx_time).total_seconds() / 3600)

                if time_diff_hours <= window_hours:
                    # Suspected infection detected
                    # Use earlier of the two times as onset
                    onset_time = min(abx_time, culture_time)
                    logger.info(f"Suspected infection detected at {onset_time}")
                    return onset_time

        return None

    def detect_organ_dysfunction(self,
                                patient_df: pd.DataFrame,
                                icu_intime: pd.Timestamp,
                                baseline_sofa: int) -> Optional[pd.Timestamp]:
        """
        Detect organ dysfunction (SOFA increase ≥2 from baseline).

        Args:
            patient_df: Patient's hourly time-series data
            icu_intime: ICU admission time
            baseline_sofa: Baseline SOFA score (minimum in first 24h)

        Returns:
            Timestamp of organ dysfunction onset, or None if no dysfunction
        """
        dysfunction_threshold = self.config['sofa']['delta_threshold']  # 2

        # Calculate SOFA for each timepoint
        for idx, row in patient_df.iterrows():
            current_sofa = self.sofa_calc.calculate_total_sofa(row)
            delta_sofa = current_sofa - baseline_sofa

            if delta_sofa >= dysfunction_threshold:
                onset_time = row['charttime']
                logger.info(f"Organ dysfunction detected at {onset_time} "
                           f"(SOFA: {baseline_sofa} → {current_sofa}, Δ={delta_sofa})")
                return onset_time

        return None

    def detect_sepsis_onset(self,
                           patient_df: pd.DataFrame,
                           prescriptions: pd.DataFrame,
                           microbiology: pd.DataFrame,
                           icu_intime: pd.Timestamp,
                           icu_outtime: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        Detect sepsis onset using Sepsis-3 definition.

        Sepsis = Suspected Infection + Organ Dysfunction (SOFA Δ ≥2)

        Args:
            patient_df: Patient's hourly time-series data
            prescriptions: Prescription data
            microbiology: Microbiology culture data
            icu_intime: ICU admission time
            icu_outtime: ICU discharge time

        Returns:
            Timestamp of sepsis onset, or None if no sepsis
        """
        # Step 1: Detect suspected infection
        infection_time = self.detect_suspected_infection(
            prescriptions, microbiology, icu_intime, icu_outtime
        )

        if infection_time is None:
            logger.info("No suspected infection detected")
            return None

        # Step 2: Calculate baseline SOFA
        baseline_sofa = self.sofa_calc.calculate_baseline_sofa(patient_df, icu_intime)

        # Step 3: Detect organ dysfunction
        dysfunction_time = self.detect_organ_dysfunction(
            patient_df, icu_intime, baseline_sofa
        )

        if dysfunction_time is None:
            logger.info("No organ dysfunction detected")
            return None

        # Sepsis onset = when both conditions are met
        # Use the later of the two timestamps
        sepsis_onset = max(infection_time, dysfunction_time)

        logger.info(f"SEPSIS DETECTED at {sepsis_onset}")
        logger.info(f"  - Infection suspected: {infection_time}")
        logger.info(f"  - Organ dysfunction: {dysfunction_time}")
        logger.info(f"  - Baseline SOFA: {baseline_sofa}")

        return sepsis_onset

    def create_labels(self,
                     patient_df: pd.DataFrame,
                     sepsis_onset: Optional[pd.Timestamp],
                     prediction_window: str = "optimal") -> pd.DataFrame:
        """
        Create labels for each timepoint in patient's ICU stay.

        Labeling strategy:
        - If sepsis occurs:
          - 6-12h before onset: POSITIVE (early prediction window)
          - After onset: POSITIVE
          - Before 12h window: NEGATIVE
        - If no sepsis: All NEGATIVE

        Args:
            patient_df: Patient's hourly data
            sepsis_onset: Sepsis onset time (None if no sepsis)
            prediction_window: "early" (12h) or "optimal" (6h)

        Returns:
            DataFrame with 'sepsis_label' and 'hours_to_onset' columns
        """
        patient_df = patient_df.copy()

        if sepsis_onset is None:
            # No sepsis: all negative
            patient_df['sepsis_label'] = 0
            patient_df['hours_to_onset'] = np.nan
            patient_df['sepsis_onset_time'] = pd.NaT
            return patient_df

        # Calculate hours until onset for each timepoint
        patient_df['sepsis_onset_time'] = sepsis_onset
        patient_df['hours_to_onset'] = (
            sepsis_onset - patient_df['charttime']
        ).dt.total_seconds() / 3600

        # Determine window thresholds
        if prediction_window == "early":
            window_start = self.prediction_window_early  # 12h
            window_end = self.prediction_window_optimal   # 6h
        else:  # "optimal"
            window_start = self.prediction_window_optimal  # 6h
            window_end = 0

        # Label timepoints
        patient_df['sepsis_label'] = 0  # Default negative

        # Positive: within prediction window OR after onset
        positive_mask = (
            (patient_df['hours_to_onset'] >= window_end) &
            (patient_df['hours_to_onset'] <= window_start)
        ) | (patient_df['hours_to_onset'] <= 0)

        patient_df.loc[positive_mask, 'sepsis_label'] = 1

        logger.debug(f"Created labels: {positive_mask.sum()} positive, "
                    f"{(~positive_mask).sum()} negative")

        return patient_df

    def label_patient(self,
                     patient_id: int,
                     patient_df: pd.DataFrame,
                     prescriptions: pd.DataFrame,
                     microbiology: pd.DataFrame,
                     icu_intime: pd.Timestamp,
                     icu_outtime: pd.Timestamp) -> Tuple[pd.DataFrame, bool]:
        """
        Complete labeling pipeline for a single patient.

        Args:
            patient_id: Patient identifier
            patient_df: Harmonized hourly time-series data
            prescriptions: Prescription data
            microbiology: Microbiology data
            icu_intime: ICU admission time
            icu_outtime: ICU discharge time

        Returns:
            Tuple of (labeled_dataframe, has_sepsis)
        """
        logger.info(f"Labeling patient {patient_id}")

        # Detect sepsis onset
        sepsis_onset = self.detect_sepsis_onset(
            patient_df, prescriptions, microbiology, icu_intime, icu_outtime
        )

        has_sepsis = sepsis_onset is not None

        # Create labels
        labeled_df = self.create_labels(patient_df, sepsis_onset)

        return labeled_df, has_sepsis


def validate_labels(labeled_df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate label distribution and statistics.

    Args:
        labeled_df: DataFrame with sepsis_label column

    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_timepoints': len(labeled_df),
        'positive_timepoints': (labeled_df['sepsis_label'] == 1).sum(),
        'negative_timepoints': (labeled_df['sepsis_label'] == 0).sum(),
        'positive_rate': (labeled_df['sepsis_label'] == 1).mean(),
        'has_sepsis_onset': labeled_df['sepsis_onset_time'].notna().any()
    }

    if stats['has_sepsis_onset']:
        onset_times = labeled_df[labeled_df['sepsis_onset_time'].notna()]['sepsis_onset_time']
        stats['sepsis_onset_time'] = onset_times.iloc[0]

        positive_df = labeled_df[labeled_df['sepsis_label'] == 1]
        if len(positive_df) > 0:
            stats['min_hours_to_onset'] = positive_df['hours_to_onset'].min()
            stats['max_hours_to_onset'] = positive_df['hours_to_onset'].max()

    return stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = {
        'prediction_window': {
            'early_hours': 12,
            'optimal_hours': 6
        },
        'infection_suspicion': {
            'antibiotic_culture_window_hours': 24
        },
        'sofa': {
            'baseline_calculation': 'minimum_first_24h',
            'delta_threshold': 2
        }
    }

    labeler = SepsisLabeler(config)
    print("SepsisLabeler ready. Use label_patient() to generate labels.")
