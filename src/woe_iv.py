"""
woe_iv.py – Step 3 of the Credit Risk Scorecard pipeline.

Computes Weight of Evidence (WoE) transformations and Information Value (IV)
for each continuous feature using the optbinning library. Features with
IV > threshold are retained as strong predictors.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)

# Features to evaluate for WoE/IV
CANDIDATE_FEATURES = [
    "fico_midpoint",
    "dti",
    "revol_util",
    "int_rate",
    "loan_to_income",
    "payment_to_income",
    "annual_inc_log",
    "emp_length_numeric",
    "loan_amnt",
    "funded_amnt",
    "installment",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "total_acc",
    "out_prncp",
    "total_pymnt",
    "last_pymnt_amnt",
    "delinq_flag",
    "term",
]


def compute_woe_iv(df: pd.DataFrame, target: str = config.TARGET_COLUMN,
                    features: list = None) -> tuple:
    """
    Compute WoE transformation and IV for each numeric feature.

    Uses OptimalBinning from the optbinning library for automatic
    monotonic binning and WoE computation.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame including the target column.
    target : str
        Name of the binary target column.
    features : list, optional
        List of feature names to process. Defaults to CANDIDATE_FEATURES.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, dict)
        - woe_df: DataFrame with WoE-transformed feature columns + target.
        - iv_table: DataFrame with columns [feature, iv] sorted descending.
        - binning_objects: dict mapping feature name → fitted OptimalBinning.
    """
    if features is None:
        features = [f for f in CANDIDATE_FEATURES if f in df.columns]

    y = df[target].values
    iv_records = []
    woe_columns = {}
    binning_objects = {}

    for feat in features:
        # Coerce to numeric, skipping columns that cannot be converted
        try:
            numeric_series = pd.to_numeric(df[feat], errors="coerce")
            x = numeric_series.values.astype(float)
        except (ValueError, TypeError):
            logger.warning("Skipping %s – cannot convert to numeric", feat)
            continue

        # Skip if all values are identical
        if np.nanstd(x) == 0:
            logger.warning("Skipping %s – zero variance", feat)
            continue

        try:
            optb = OptimalBinning(
                name=feat,
                dtype="numerical",
                solver="cp",
                monotonic_trend="auto",
            )
            optb.fit(x, y)

            # Extract IV from the binning table
            binning_table = optb.binning_table
            iv_value = binning_table.build()["IV"].values[-1]  # Totals row

            iv_records.append({"feature": feat, "iv": iv_value})
            binning_objects[feat] = optb

            # WoE-transformed column
            woe_columns[f"{feat}_woe"] = optb.transform(x, metric="woe")

            logger.info("%-25s IV = %.4f", feat, iv_value)

        except Exception as exc:
            logger.warning("Could not bin %s: %s", feat, exc)

    # Build IV summary table
    iv_table = (
        pd.DataFrame(iv_records)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )

    # Build WoE DataFrame
    woe_df = pd.DataFrame(woe_columns)
    woe_df[target] = y

    return woe_df, iv_table, binning_objects


def filter_by_iv(woe_df: pd.DataFrame, iv_table: pd.DataFrame,
                  threshold: float = config.IV_THRESHOLD) -> tuple:
    """
    Retain only WoE features whose IV exceeds the threshold.

    Parameters
    ----------
    woe_df : pd.DataFrame
        Full WoE-transformed DataFrame.
    iv_table : pd.DataFrame
        IV summary table.
    threshold : float
        Minimum IV to keep (default from config).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - Filtered WoE DataFrame (selected features + target).
        - Filtered IV table.
    """
    strong = iv_table[iv_table["iv"] > threshold]["feature"].tolist()
    keep_cols = [f"{f}_woe" for f in strong if f"{f}_woe" in woe_df.columns]
    keep_cols.append(config.TARGET_COLUMN)
    filtered_woe = woe_df[keep_cols].copy()

    filtered_iv = iv_table[iv_table["iv"] > threshold].reset_index(drop=True)
    logger.info(
        "Retained %d features with IV > %.2f: %s",
        len(strong), threshold, strong,
    )
    return filtered_woe, filtered_iv


def save_outputs(woe_df: pd.DataFrame, iv_table: pd.DataFrame) -> None:
    """
    Save WoE-transformed dataset and IV summary to CSV files.

    Parameters
    ----------
    woe_df : pd.DataFrame
        WoE-transformed DataFrame.
    iv_table : pd.DataFrame
        IV summary DataFrame.
    """
    config.DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

    woe_df.to_csv(config.WOE_FEATURES_CSV, index=False)
    logger.info("Saved WoE features to %s", config.WOE_FEATURES_CSV)

    iv_table.to_csv(config.IV_SUMMARY_CSV, index=False)
    logger.info("Saved IV summary to %s", config.IV_SUMMARY_CSV)


def run() -> tuple:
    """
    Execute the WoE/IV pipeline end to end.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - Filtered WoE DataFrame.
        - Filtered IV table.
    """
    df = pd.read_csv(config.FEATURES_CSV)
    woe_df, iv_table, _ = compute_woe_iv(df)
    filtered_woe, filtered_iv = filter_by_iv(woe_df, iv_table)
    save_outputs(filtered_woe, filtered_iv)
    return filtered_woe, filtered_iv


if __name__ == "__main__":
    woe_features, iv_summary = run()
    print("\n─── Information Value Summary ───")
    print(iv_summary.to_string(index=False))
    print(f"\nWoE feature matrix shape: {woe_features.shape}")
