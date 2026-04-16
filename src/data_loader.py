"""
data_loader.py – Step 1 of the Credit Risk Scorecard pipeline.

Loads the raw Lending Club CSV, filters to Fully Paid and Charged Off loans,
creates a binary target, handles missing values, and saves a cleaned dataset.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ─── Add project root to path so config is importable ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)


def load_raw_data(filepath: Path = config.RAW_CSV) -> pd.DataFrame:
    """
    Load the raw Lending Club CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : Path
        Absolute path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame as read from disk.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the specified path.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found at '{filepath}'.\n"
            "Please download the Lending Club Loan Data (2007-2018) from Kaggle "
            "and place it at data/raw/loan.csv."
        )
    logger.info("Loading raw data from %s", filepath)
    df = pd.read_csv(filepath, low_memory=False)
    logger.info("Raw data shape: %s", df.shape)
    return df


def filter_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where loan_status is 'Fully Paid' or 'Charged Off'.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    mask = df["loan_status"].isin(config.KEEP_LOAN_STATUS)
    df_filtered = df.loc[mask].copy()
    logger.info(
        "Filtered to %d rows (Fully Paid + Charged Off)", len(df_filtered)
    )
    return df_filtered


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target column `bad_loan`.

    1 = Charged Off (default), 0 = Fully Paid.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a `loan_status` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional `bad_loan` column.
    """
    df[config.TARGET_COLUMN] = (df["loan_status"] == "Charged Off").astype(int)
    class_balance = df[config.TARGET_COLUMN].value_counts(normalize=True)
    logger.info("Class balance:\n%s", class_balance.to_string())
    return df


def select_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset the DataFrame to the key columns listed in config.

    Columns that do not exist in the DataFrame are silently ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the key columns (+ target).
    """
    cols = [c for c in config.KEY_COLUMNS if c in df.columns]
    # Always keep target
    if config.TARGET_COLUMN in df.columns and config.TARGET_COLUMN not in cols:
        cols.append(config.TARGET_COLUMN)
    df = df[cols].copy()
    logger.info("Selected %d key columns", len(cols))
    return df


def drop_high_missing(df: pd.DataFrame, threshold: float = config.MISSING_THRESHOLD) -> pd.DataFrame:
    """
    Drop columns with more than `threshold` fraction of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float
        Maximum allowed fraction of NaN values (default 0.40).

    Returns
    -------
    pd.DataFrame
        DataFrame with high-missing columns removed.
    """
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    if drop_cols:
        logger.info("Dropping %d columns with >%.0f%% missing: %s",
                     len(drop_cols), threshold * 100, drop_cols)
    df = df.drop(columns=drop_cols)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values: median for numeric columns, mode for categorical.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential NaN values.

    Returns
    -------
    pd.DataFrame
        DataFrame with NaNs imputed.
    """
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Filled %s (numeric) with median %.4f", col, median_val)

    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug("Filled %s (categorical) with mode '%s'", col, mode_val)

    logger.info("Imputation complete – remaining NaNs: %d", df.isnull().sum().sum())
    return df


def save_cleaned(df: pd.DataFrame, filepath: Path = config.CLEANED_CSV) -> None:
    """
    Save the cleaned DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    filepath : Path
        Destination path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Saved cleaned data to %s", filepath)


def run() -> pd.DataFrame:
    """
    Execute the full data-loading pipeline.

    Returns
    -------
    pd.DataFrame
        Cleaned and saved DataFrame.
    """
    df = load_raw_data()
    df = filter_loan_status(df)
    df = create_target(df)
    df = select_key_columns(df)
    df = drop_high_missing(df)
    df = impute_missing(df)
    save_cleaned(df)
    logger.info("Final cleaned shape: %s", df.shape)
    return df


if __name__ == "__main__":
    cleaned = run()
    print(f"\nData loaded: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
    print(f"Class balance:\n{cleaned[config.TARGET_COLUMN].value_counts()}")
