"""
scorecard.py – Step 4 of the Credit Risk Scorecard pipeline.

Trains a Logistic Regression model on WoE-transformed features and converts
predicted probabilities into a credit score on a 300-900 scale using PDO
(Points to Double Odds) scaling. Assigns risk tiers to every applicant.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)


def train_logistic_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train a Logistic Regression classifier on WoE features.

    Parameters
    ----------
    X : pd.DataFrame
        WoE-transformed feature matrix.
    y : pd.Series
        Binary target (1 = default, 0 = fully paid).

    Returns
    -------
    tuple of (LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        - Fitted model
        - X_train, X_test, y_train, y_test arrays
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression trained – coefficients: %s", model.coef_[0])
    logger.info("Intercept: %.4f", model.intercept_[0])

    return model, X_train, X_test, y_train, y_test


def compute_raw_score(predicted_proba: np.ndarray) -> np.ndarray:
    """
    Convert predicted default probabilities into raw credit scores using
    the PDO (Points to Double Odds) methodology.

    Formula:
        odds = (1 - p) / p
        Score = BASE_SCORE + PDO * log2(odds / BASE_ODDS)

    Parameters
    ----------
    predicted_proba : np.ndarray
        Predicted probability of default (bad_loan = 1) for each applicant.

    Returns
    -------
    np.ndarray
        Raw (unscaled) credit scores.
    """
    # Clip probabilities to avoid log(0) or division by zero
    proba = np.clip(predicted_proba, 1e-6, 1 - 1e-6)
    odds = (1 - proba) / proba
    raw_score = config.BASE_SCORE + config.PDO * np.log2(odds / config.BASE_ODDS)
    return raw_score


def scale_scores(raw_scores: np.ndarray) -> np.ndarray:
    """
    Scale raw credit scores to the [SCORE_MIN, SCORE_MAX] range.

    Parameters
    ----------
    raw_scores : np.ndarray
        Unscaled credit scores.

    Returns
    -------
    np.ndarray
        Scores mapped to [300, 900].
    """
    scaler = MinMaxScaler(feature_range=(config.SCORE_MIN, config.SCORE_MAX))
    scaled = scaler.fit_transform(raw_scores.reshape(-1, 1)).ravel()
    return scaled


def assign_risk_tier(score: float) -> str:
    """
    Map a numeric credit score to a named risk tier.

    Parameters
    ----------
    score : float
        Credit score (300-900).

    Returns
    -------
    str
        Risk tier label.
    """
    for tier, (low, high) in config.RISK_TIERS.items():
        if low <= score <= high:
            return tier
    return "Deep Subprime"  # Fallback


def build_scorecard(woe_df: pd.DataFrame, features_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Execute the full scoring pipeline: train model → compute scores → assign tiers.

    Parameters
    ----------
    woe_df : pd.DataFrame
        WoE-transformed features + target column.
    features_df : pd.DataFrame, optional
        Original features DataFrame to join with scores for the final output.

    Returns
    -------
    pd.DataFrame
        Scored DataFrame with predicted_proba, credit_score, and risk_tier.
    """
    target = config.TARGET_COLUMN
    feature_cols = [c for c in woe_df.columns if c != target]
    X = woe_df[feature_cols]
    y = woe_df[target]

    model, X_train, X_test, y_train, y_test = train_logistic_model(X, y)

    # Predict on full dataset
    predicted_proba = model.predict_proba(X)[:, 1]
    raw_scores = compute_raw_score(predicted_proba)
    credit_scores = scale_scores(raw_scores)

    # Build output DataFrame
    output = pd.DataFrame()
    if features_df is not None and len(features_df) == len(woe_df):
        output = features_df.copy()
    else:
        output = woe_df.copy()

    output["predicted_proba"] = predicted_proba
    output["credit_score"] = np.round(credit_scores).astype(int)
    output["risk_tier"] = output["credit_score"].apply(assign_risk_tier)

    # Score distribution
    tier_dist = output["risk_tier"].value_counts()
    logger.info("Risk tier distribution:\n%s", tier_dist.to_string())
    logger.info("Score stats: mean=%.1f, std=%.1f, min=%d, max=%d",
                output["credit_score"].mean(), output["credit_score"].std(),
                output["credit_score"].min(), output["credit_score"].max())

    return output, model, X_test, y_test


def save_scorecard(output: pd.DataFrame, filepath: Path = config.SCORECARD_OUTPUT_CSV) -> None:
    """
    Save the final scored dataset to CSV.

    Parameters
    ----------
    output : pd.DataFrame
        Scored DataFrame.
    filepath : Path
        Destination path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(filepath, index=False)
    logger.info("Saved scorecard output to %s (%d rows)", filepath, len(output))


def run() -> tuple:
    """
    Execute the scorecard pipeline.

    Returns
    -------
    tuple of (pd.DataFrame, sklearn model, X_test, y_test)
        Scored DataFrame and model artefacts for evaluation.
    """
    woe_df = pd.read_csv(config.WOE_FEATURES_CSV)

    # Try to load the original features for richer output
    features_df = None
    if config.FEATURES_CSV.exists():
        features_df = pd.read_csv(config.FEATURES_CSV)

    output, model, X_test, y_test = build_scorecard(woe_df, features_df)
    save_scorecard(output)
    return output, model, X_test, y_test


if __name__ == "__main__":
    scored_df, _, _, _ = run()
    print(f"\nScorecard generated: {len(scored_df)} rows")
    print(f"Score range: {scored_df['credit_score'].min()} – {scored_df['credit_score'].max()}")
    print(f"\nRisk tier distribution:\n{scored_df['risk_tier'].value_counts().to_string()}")
