"""
Data Harmonization: MIMIC-IV → PhysioNet CinC 2019 Schema

This module maps MIMIC-IV's complex schema (itemids, irregular timestamps) to
CinC 2019's flat hourly schema (40 canonical variables).

Key Functions:
- Variable mapping (itemid → CinC variable name)
- Unit conversion (Temperature F→C, Lactate mg/dL→mmol/L)
- Temporal alignment (irregular → hourly bins)

Author: Jason
Date: January 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
import logging

logger = logging.getLogger(__name__)


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
        self.temporal_config = self.config['temporal_alignment']

        # Create reverse mapping: itemid -> variable name
        self.itemid_to_variable = self._create_reverse_mapping()

        logger.info(f"MIMICHarmonizer initialized with {len(self.variable_mapping)} variables")

    def _create_reverse_mapping(self) -> Dict[int, str]:
        """
        Create reverse mapping from itemid to variable name.

        Returns:
            Dictionary mapping itemid -> variable_name
        """
        reverse_map = {}
        for var_name, itemids in self.variable_mapping.items():
            for itemid in itemids:
                reverse_map[itemid] = var_name

        return reverse_map

    def map_chartevents(self, chartevents: pd.DataFrame) -> pd.DataFrame:
        """
        Map chartevents itemids to CinC canonical variables.

        Args:
            chartevents: MIMIC-IV chartevents DataFrame
                Required columns: subject_id, hadm_id, charttime, itemid, value, valueuom

        Returns:
            DataFrame with columns: subject_id, hadm_id, charttime, variable, value
        """
        logger.info(f"Mapping chartevents: {len(chartevents)} rows")

        # Filter to only itemids we care about
        relevant_itemids = set(self.itemid_to_variable.keys())
        chartevents_filtered = chartevents[chartevents['itemid'].isin(relevant_itemids)].copy()

        logger.info(f"Filtered to {len(chartevents_filtered)} relevant rows")

        # Map itemid to variable name
        chartevents_filtered['variable'] = chartevents_filtered['itemid'].map(
            self.itemid_to_variable
        )

        # Convert value to numeric, coercing errors to NaN
        chartevents_filtered['value'] = pd.to_numeric(chartevents_filtered['value'], errors='coerce')

        # Select and rename columns
        result = chartevents_filtered[[
            'subject_id', 'hadm_id', 'charttime', 'variable', 'value', 'valueuom'
        ]]

        return result

    def map_labevents(self, labevents: pd.DataFrame) -> pd.DataFrame:
        """
        Map labevents itemids to CinC canonical variables.

        Args:
            labevents: MIMIC-IV labevents DataFrame
                Required columns: subject_id, hadm_id, charttime, itemid, value, valueuom

        Returns:
            DataFrame with columns: subject_id, hadm_id, charttime, variable, value
        """
        logger.info(f"Mapping labevents: {len(labevents)} rows")

        # Filter to only itemids we care about
        relevant_itemids = set(self.itemid_to_variable.keys())
        labevents_filtered = labevents[labevents['itemid'].isin(relevant_itemids)].copy()

        logger.info(f"Filtered to {len(labevents_filtered)} relevant rows")

        # Map itemid to variable name
        labevents_filtered['variable'] = labevents_filtered['itemid'].map(
            self.itemid_to_variable
        )

        # Convert value to numeric, coercing errors to NaN
        labevents_filtered['value'] = pd.to_numeric(labevents_filtered['value'], errors='coerce')

        # Select and rename columns
        result = labevents_filtered[[
            'subject_id', 'hadm_id', 'charttime', 'variable', 'value', 'valueuom'
        ]]

        return result

    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply unit conversions to ensure consistency.

        Conversions:
        - Temperature: Fahrenheit → Celsius
        - FiO2: Percentage → Fraction (if needed)

        Args:
            df: DataFrame with 'variable', 'value', and 'valueuom' columns

        Returns:
            DataFrame with standardized units
        """
        df = df.copy()

        # Temperature conversion: F → C
        if self.unit_conversions['temperature']['enabled']:
            temp_mask = (df['variable'] == 'Temp') & \
                       (df['valueuom'].str.upper().isin(['F', '°F', 'FAHRENHEIT']))

            if temp_mask.sum() > 0:
                logger.info(f"Converting {temp_mask.sum()} temperature values from F to C")
                df.loc[temp_mask, 'value'] = (df.loc[temp_mask, 'value'] - 32) * 5/9
                df.loc[temp_mask, 'valueuom'] = 'C'

        # FiO2 conversion: Percentage to fraction
        fio2_mask = (df['variable'] == 'FiO2') & (df['value'] > 1)
        if fio2_mask.sum() > 0:
            logger.info(f"Converting {fio2_mask.sum()} FiO2 values from % to fraction")
            df.loc[fio2_mask, 'value'] = df.loc[fio2_mask, 'value'] / 100.0

        return df

    def create_hourly_bins(self,
                          df: pd.DataFrame,
                          subject_id: int,
                          icu_intime: pd.Timestamp,
                          icu_outtime: pd.Timestamp) -> pd.DataFrame:
        """
        Align irregular timestamps to hourly bins.

        MIMIC-IV has observations at irregular intervals (every few minutes).
        CinC 2019 expects hourly observations.

        Strategy:
        - Create 1-hour bins from ICU admission
        - Aggregate vitals: median (robust to outliers)
        - Aggregate labs: last (most recent value)
        - Aggregate urine output: sum (total in hour)

        Args:
            df: Patient data with irregular timestamps
            subject_id: Patient identifier
            icu_intime: ICU admission timestamp
            icu_outtime: ICU discharge timestamp

        Returns:
            DataFrame with hourly observations
        """
        logger.debug(f"Creating hourly bins for patient {subject_id}")

        # Filter to ICU stay period
        df_icu = df[
            (df['charttime'] >= icu_intime) &
            (df['charttime'] <= icu_outtime)
        ].copy()

        if len(df_icu) == 0:
            logger.warning(f"No data for patient {subject_id} during ICU stay")
            return pd.DataFrame()

        # Create hourly bins
        df_icu['hour_bin'] = df_icu['charttime'].dt.floor('H')

        # Define aggregation strategy per variable
        agg_vitals = self.temporal_config['aggregation']['vitals']  # 'median'
        agg_labs = self.temporal_config['aggregation']['labs']      # 'last'
        agg_urine = self.temporal_config['aggregation']['urine_output']  # 'sum'

        # Separate variables by aggregation type
        vitals = ['HR', 'Resp', 'Temp', 'SBP', 'MAP', 'DBP', 'O2Sat']
        labs = ['Lactate', 'Creatinine', 'Bilirubin_total', 'Platelets', 'WBC',
                'Glucose', 'BUN', 'Potassium', 'Sodium', 'Chloride', 'Magnesium',
                'Calcium', 'Ionized_calcium', 'Bicarbonate', 'pH', 'PaO2', 'PaCO2',
                'Base_excess', 'FiO2']
        urine_vars = ['Urine_output']

        hourly_data = []

        for hour in df_icu['hour_bin'].unique():
            hour_df = df_icu[df_icu['hour_bin'] == hour]

            hour_values = {
                'subject_id': subject_id,
                'charttime': hour
            }

            # Aggregate vitals (median)
            for var in vitals:
                var_data = hour_df[hour_df['variable'] == var]['value']
                if len(var_data) > 0:
                    hour_values[var] = var_data.median()

            # Aggregate labs (last/most recent)
            for var in labs:
                var_data = hour_df[hour_df['variable'] == var].sort_values('charttime')
                if len(var_data) > 0:
                    hour_values[var] = var_data['value'].iloc[-1]

            # Aggregate urine output (sum)
            for var in urine_vars:
                var_data = hour_df[hour_df['variable'] == var]['value']
                if len(var_data) > 0:
                    hour_values[var] = var_data.sum()

            hourly_data.append(hour_values)

        result = pd.DataFrame(hourly_data)
        logger.debug(f"Created {len(result)} hourly observations for patient {subject_id}")

        return result

    def forward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply forward fill with time-based limits.

        Forward fill missing values up to specified time limits:
        - Vitals: 6 hours
        - Labs: 24 hours
        - GCS: 12 hours

        Args:
            df: DataFrame with hourly data

        Returns:
            DataFrame with forward-filled values
        """
        if len(df) == 0:
            return df

        df = df.copy().sort_values('charttime')

        ff_config = self.temporal_config['forward_fill']
        if not ff_config['enabled']:
            return df

        max_hours_vitals = ff_config['max_hours']['vitals']
        max_hours_labs = ff_config['max_hours']['labs']
        max_hours_gcs = ff_config['max_hours']['gcs']

        vitals = ['HR', 'Resp', 'Temp', 'SBP', 'MAP', 'DBP', 'O2Sat']
        labs = ['Lactate', 'Creatinine', 'Bilirubin_total', 'Platelets', 'WBC',
                'Glucose', 'BUN', 'Potassium', 'Sodium', 'Chloride', 'Magnesium',
                'Calcium', 'Ionized_calcium', 'Bicarbonate', 'pH', 'PaO2', 'PaCO2',
                'Base_excess', 'FiO2']
        gcs_vars = ['GCS']

        # Forward fill vitals (limit 6 hours)
        for var in vitals:
            if var in df.columns:
                df[var] = df[var].fillna(method='ffill', limit=max_hours_vitals)

        # Forward fill labs (limit 24 hours)
        for var in labs:
            if var in df.columns:
                df[var] = df[var].fillna(method='ffill', limit=max_hours_labs)

        # Forward fill GCS (limit 12 hours)
        for var in gcs_vars:
            if var in df.columns:
                df[var] = df[var].fillna(method='ffill', limit=max_hours_gcs)

        return df

    def harmonize_patient(self,
                         subject_id: int,
                         chartevents: pd.DataFrame,
                         labevents: pd.DataFrame,
                         icu_intime: pd.Timestamp,
                         icu_outtime: pd.Timestamp) -> pd.DataFrame:
        """
        Complete harmonization pipeline for a single patient.

        Args:
            subject_id: Patient ID
            chartevents: Patient's chartevents data
            labevents: Patient's labevents data
            icu_intime: ICU admission time
            icu_outtime: ICU discharge time

        Returns:
            DataFrame with hourly CinC-formatted data
        """
        # Map itemids to variables
        chart_mapped = self.map_chartevents(chartevents)
        lab_mapped = self.map_labevents(labevents)

        # Combine chartevents and labevents
        combined = pd.concat([chart_mapped, lab_mapped], ignore_index=True)

        # Convert units
        combined = self.convert_units(combined)

        # Create hourly bins
        hourly = self.create_hourly_bins(combined, subject_id, icu_intime, icu_outtime)

        # Forward fill missing values
        hourly = self.forward_fill(hourly)

        # Rename columns to match SOFA calculator expectations
        hourly = self.rename_for_sofa(hourly)

        return hourly

    def rename_for_sofa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to match SOFA calculator expectations.

        The SOFA calculator expects lowercase column names:
        - 'pao2', 'fio2', 'platelets', 'bilirubin', etc.

        But we create CamelCase names from config:
        - 'PaO2', 'FiO2', 'Platelets', 'Bilirubin_total', etc.

        Args:
            df: DataFrame with CamelCase column names

        Returns:
            DataFrame with renamed columns
        """
        if len(df) == 0:
            return df

        rename_mapping = {
            'PaO2': 'pao2',
            'FiO2': 'fio2',
            'Platelets': 'platelets',
            'Bilirubin_total': 'bilirubin',
            'MAP': 'map_value',
            'GCS': 'gcs',
            'Creatinine': 'creatinine',
            'Urine_output': 'urine_output',
            'HR': 'hr',
            'O2Sat': 'o2sat',
            'Temp': 'temp',
            'SBP': 'sbp',
            'DBP': 'dbp',
            'Resp': 'resp',
            'Lactate': 'lactate',
            'WBC': 'wbc',
            'Glucose': 'glucose',
            'BUN': 'bun',
            'Potassium': 'potassium',
            'Sodium': 'sodium',
            'Chloride': 'chloride',
            'Magnesium': 'magnesium',
            'Calcium': 'calcium',
            'Ionized_calcium': 'ionized_calcium',
            'Bicarbonate': 'bicarbonate',
            'pH': 'ph',
            'PaCO2': 'paco2',
            'Base_excess': 'base_excess',
            'Mechanical_ventilation': 'is_ventilated'
        }

        # Only rename columns that exist
        existing_renames = {k: v for k, v in rename_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_renames)

        logger.debug(f"Renamed {len(existing_renames)} columns for SOFA compatibility")

        return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    harmonizer = MIMICHarmonizer("config/data_config.yaml")

    # Example: Load and harmonize data
    # chartevents = pd.read_csv("data/raw/mimic_iv/chartevents.csv.gz", nrows=10000)
    # labevents = pd.read_csv("data/raw/mimic_iv/labevents.csv.gz", nrows=10000)
    # result = harmonizer.harmonize_patient(...)

    print("Harmonizer ready. Use harmonize_patient() to process data.")
