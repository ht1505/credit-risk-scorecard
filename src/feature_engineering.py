"""
feature_engineering.py – Step 2 of the Credit Risk Scorecard pipeline.

Creates derived features expected by the WoE/IV module:
  - Payment-to-income ratio
  - Delinquency flag
  - FICO midpoint
  - Loan-to-income ratio
  - Employment length (numeric)
  - Income log transform
  - DTI outlier capping
  - Interest rate numeric conversion
"""

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)


def cap_outliers(series: pd.Series, upper_pct: float = 99) -> pd.Series:
    """
    Cap values above the `upper_pct` percentile.

    Parameters
    ----------
    series : pd.Series
        Numeric series to cap.
    upper_pct : float
        Percentile threshold (default 99).

    Returns
    -------
    pd.Series
        Capped series.
    """
    cap_val = np.percentile(series.dropna(), upper_pct)
    capped = series.clip(upper=cap_val)
    logger.debug("Capped %s at %.4f (p%d)", series.name, cap_val, upper_pct)
    return capped


def convert_int_rate(series: pd.Series) -> pd.Series:
    """
    Strip '%' from interest rate strings and convert to float.

    If the column is already numeric, it is returned as-is.

    Parameters
    ----------
    series : pd.Series
        Interest rate column, possibly with '%' suffix.

    Returns
    -------
    pd.Series
        Numeric interest rate values.
    """
    if series.dtype == object:
        series = series.astype(str).str.replace("%", "", regex=False).astype(float)
        logger.info("Converted int_rate from string to float")
    return series


def convert_emp_length(series: pd.Series) -> pd.Series:
    """
    Convert employment length strings to numeric (0-10 scale).

    Mapping:
      - '< 1 year' → 0
      - '1 year' → 1
      - '2 years' → 2  ...  '9 years' → 9
      - '10+ years' → 10
      - NaN / unrecognised → 0

    Parameters
    ----------
    series : pd.Series
        Employment length as string.

    Returns
    -------
    pd.Series
        Numeric employment length (0–10).
    """
    def _parse(val):
        if pd.isna(val):
            return 0
        val = str(val).strip()
        if val.startswith("< 1") or val.startswith("<1"):
            return 0
        if val.startswith("10"):
            return 10
        m = re.search(r"(\d+)", val)
        return int(m.group(1)) if m else 0

    result = series.apply(_parse).astype(int)
    logger.info("Converted emp_length to numeric (0-10)")
    return result


def convert_term(series: pd.Series) -> pd.Series:
    """
    Convert term strings like ' 36 months' to integer months.

    Parameters
    ----------
    series : pd.Series
        Term column.

    Returns
    -------
    pd.Series
        Integer term in months.
    """
    if series.dtype == object:
        series = series.astype(str).str.extract(r"(\d+)")[0].astype(float)
        logger.info("Converted term to numeric months")
    return series


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from data_loader.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features appended.
    """
    logger.info("Starting feature engineering on %d rows", len(df))

    # Interest rate: strip % and convert
    if "int_rate" in df.columns:
        df["int_rate"] = convert_int_rate(df["int_rate"])

    # Term: convert to numeric
    if "term" in df.columns:
        df["term"] = convert_term(df["term"])

    # DTI: cap outliers at 99th percentile
    if "dti" in df.columns:
        df["dti"] = cap_outliers(df["dti"], upper_pct=99)

    # Payment-to-income ratio
    if "installment" in df.columns and "annual_inc" in df.columns:
        monthly_inc = df["annual_inc"] / 12
        # Avoid division by zero
        monthly_inc = monthly_inc.replace(0, np.nan)
        df["payment_to_income"] = df["installment"] / monthly_inc
        df["payment_to_income"] = df["payment_to_income"].fillna(0)
        logger.info("Created payment_to_income")

    # Delinquency flag
    if "delinq_2yrs" in df.columns:
        df["delinq_flag"] = (df["delinq_2yrs"] > 0).astype(int)
        logger.info("Created delinq_flag")

    # FICO midpoint
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_midpoint"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        logger.info("Created fico_midpoint")

    # Loan-to-income ratio
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        annual_inc_safe = df["annual_inc"].replace(0, np.nan)
        df["loan_to_income"] = df["loan_amnt"] / annual_inc_safe
        df["loan_to_income"] = df["loan_to_income"].fillna(0)
        logger.info("Created loan_to_income")

    # Employment length numeric
    if "emp_length" in df.columns:
        df["emp_length_numeric"] = convert_emp_length(df["emp_length"])

    # Income log transform
    if "annual_inc" in df.columns:
        df["annual_inc_log"] = np.log1p(df["annual_inc"])
        logger.info("Created annual_inc_log")

    logger.info("Feature engineering complete – %d columns", df.shape[1])
    return df


def save_features(df: pd.DataFrame, filepath: Path = config.FEATURES_CSV) -> None:
    """
    Save the engineered-features DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame.
    filepath : Path
        Destination path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Saved engineered features to %s", filepath)


def run() -> pd.DataFrame:
    """
    Execute the feature engineering pipeline.

    Reads cleaned data, engineers features, saves to disk.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features.
    """
    df = pd.read_csv(config.CLEANED_CSV)
    df = engineer_features(df)
    save_features(df)
    return df


if __name__ == "__main__":
    features_df = run()
    print(f"\nFeatures engineered: {features_df.shape[1]} columns, {features_df.shape[0]} rows")
    print(f"Columns: {list(features_df.columns)}")
