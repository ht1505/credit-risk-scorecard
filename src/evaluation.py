"""
evaluation.py – Step 5 of the Credit Risk Scorecard pipeline.

Computes model performance metrics and generates diagnostic plots:
  - KS Statistic (Kolmogorov-Smirnov) with curve
  - Gini Coefficient
  - ROC-AUC with curve
  - Decile analysis table
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)

# Ensure the plots directory exists
config.PLOTS_PATH.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# KS Statistic
# ──────────────────────────────────────────────────────────────────────────────

def compute_ks(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute the Kolmogorov-Smirnov statistic.

    The KS statistic measures the maximum separation between the cumulative
    distribution functions of the predicted probabilities for good and bad
    borrowers.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1).
    y_proba : np.ndarray
        Predicted probability of default.

    Returns
    -------
    float
        KS statistic value (0 to 1).
    """
    good_proba = y_proba[y_true == 0]
    bad_proba = y_proba[y_true == 1]
    ks_stat, _ = sp_stats.ks_2samp(bad_proba, good_proba)
    logger.info("KS Statistic: %.4f", ks_stat)
    return ks_stat


def plot_ks_curve(y_true: np.ndarray, y_proba: np.ndarray,
                  save_path: Path = config.PLOTS_PATH / "ks_curve.png") -> float:
    """
    Plot the KS (Kolmogorov-Smirnov) curve and save to disk.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probability of default.
    save_path : Path
        File path for the saved plot.

    Returns
    -------
    float
        KS statistic value.
    """
    thresholds = np.linspace(0, 1, 200)
    good_cdf = np.array([(y_proba[y_true == 0] <= t).mean() for t in thresholds])
    bad_cdf = np.array([(y_proba[y_true == 1] <= t).mean() for t in thresholds])
    ks_values = np.abs(bad_cdf - good_cdf)
    ks_stat = ks_values.max()
    ks_threshold = thresholds[np.argmax(ks_values)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, good_cdf, label="Good (Fully Paid)", color="#2ecc71", linewidth=2)
    ax.plot(thresholds, bad_cdf, label="Bad (Charged Off)", color="#e74c3c", linewidth=2)
    ax.fill_between(thresholds, good_cdf, bad_cdf, alpha=0.15, color="#3498db")

    # Annotate KS point
    ax.axvline(x=ks_threshold, linestyle="--", color="#7f8c8d", linewidth=1)
    ax.annotate(
        f"KS = {ks_stat:.4f}",
        xy=(ks_threshold, (good_cdf[np.argmax(ks_values)] + bad_cdf[np.argmax(ks_values)]) / 2),
        fontsize=13, fontweight="bold", color="#2c3e50",
        xytext=(ks_threshold + 0.05, 0.5),
        arrowprops=dict(arrowstyle="->", color="#2c3e50"),
    )

    ax.set_title("KS Curve – Good vs Bad Borrowers", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Probability Threshold", fontsize=12)
    ax.set_ylabel("Cumulative Distribution", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=config.PLOT_DPI)
    plt.close(fig)
    logger.info("KS curve saved to %s", save_path)
    return ks_stat


# ──────────────────────────────────────────────────────────────────────────────
# ROC-AUC
# ──────────────────────────────────────────────────────────────────────────────

def compute_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute the Area Under the ROC Curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probability of default.

    Returns
    -------
    float
        AUC score.
    """
    auc = roc_auc_score(y_true, y_proba)
    logger.info("ROC-AUC: %.4f", auc)
    return auc


def compute_gini(auc: float) -> float:
    """
    Compute the Gini coefficient from AUC.

    Gini = 2 * AUC - 1

    Parameters
    ----------
    auc : float
        Area Under the ROC Curve.

    Returns
    -------
    float
        Gini coefficient.
    """
    gini = 2 * auc - 1
    logger.info("Gini Coefficient: %.4f", gini)
    return gini


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   save_path: Path = config.PLOTS_PATH / "roc_curve.png") -> float:
    """
    Plot the ROC curve with AUC annotation and save to disk.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probability of default.
    save_path : Path
        File path for the saved plot.

    Returns
    -------
    float
        AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, color="#3498db", linewidth=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#3498db")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#95a5a6", linewidth=1, label="Random Classifier")

    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=config.PLOT_DPI)
    plt.close(fig)
    logger.info("ROC curve saved to %s", save_path)
    return auc


# ──────────────────────────────────────────────────────────────────────────────
# Decile Analysis
# ──────────────────────────────────────────────────────────────────────────────

def decile_table(y_true: np.ndarray, credit_scores: np.ndarray) -> pd.DataFrame:
    """
    Build a decile analysis table sorted by credit score descending.

    Each decile shows: count, number of bads, bad rate,
    cumulative bad rate, and cumulative percentage of bads captured.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    credit_scores : np.ndarray
        Numeric credit scores.

    Returns
    -------
    pd.DataFrame
        Decile analysis table.
    """
    tmp = pd.DataFrame({"score": credit_scores, "bad": y_true})
    tmp = tmp.sort_values("score", ascending=False).reset_index(drop=True)
    tmp["decile"] = pd.qcut(tmp.index, 10, labels=False, duplicates="drop") + 1

    agg = tmp.groupby("decile").agg(
        count=("bad", "size"),
        bads=("bad", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index()

    agg["bad_rate"] = (agg["bads"] / agg["count"] * 100).round(2)
    agg["cum_bads"] = agg["bads"].cumsum()
    agg["cum_bad_rate"] = (agg["cum_bads"] / agg["bads"].sum() * 100).round(2)

    logger.info("Decile table:\n%s", agg.to_string(index=False))
    return agg


def plot_decile_chart(decile_df: pd.DataFrame,
                      save_path: Path = config.PLOTS_PATH / "decile_chart.png") -> None:
    """
    Plot a bar chart of bad rate by decile and save to disk.

    Parameters
    ----------
    decile_df : pd.DataFrame
        Output from decile_table().
    save_path : Path
        File path for the saved plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(
        decile_df["decile"],
        decile_df["bad_rate"],
        color="#e74c3c",
        alpha=0.75,
        edgecolor="#c0392b",
        label="Bad Rate (%)",
    )
    ax1.set_xlabel("Decile (1 = Best Score)", fontsize=12)
    ax1.set_ylabel("Bad Rate (%)", fontsize=12, color="#e74c3c")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, height + 0.3,
            f"{height:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Cumulative bad rate line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        decile_df["decile"],
        decile_df["cum_bad_rate"],
        color="#2ecc71",
        marker="o",
        linewidth=2,
        label="Cumulative Bad Capture (%)",
    )
    ax2.set_ylabel("Cumulative Bad Capture (%)", fontsize=12, color="#2ecc71")
    ax2.tick_params(axis="y", labelcolor="#2ecc71")

    ax1.set_title("Decile Analysis – Bad Rate & Cumulative Bad Capture",
                   fontsize=14, fontweight="bold")
    ax1.set_xticks(decile_df["decile"])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=config.PLOT_DPI)
    plt.close(fig)
    logger.info("Decile chart saved to %s", save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(ks: float, gini: float, auc: float,
                  scored_df: pd.DataFrame) -> None:
    """
    Print a formatted summary of all evaluation metrics.

    Parameters
    ----------
    ks : float
        KS statistic.
    gini : float
        Gini coefficient.
    auc : float
        ROC-AUC score.
    scored_df : pd.DataFrame
        Scored DataFrame with credit_score and bad_loan columns.
    """
    # Approval rate at cutoff
    above_cutoff = (scored_df["credit_score"] >= config.APPROVAL_CUTOFF).mean() * 100

    print("\n" + "=" * 55)
    print("        MODEL EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  KS Statistic          : {ks:.4f}")
    print(f"  Gini Coefficient      : {gini:.4f}")
    print(f"  ROC-AUC               : {auc:.4f}")
    print(f"  Approval Rate (>={config.APPROVAL_CUTOFF}) : {above_cutoff:.1f}%")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ──────────────────────────────────────────────────────────────────────────────

def run(scored_df: pd.DataFrame = None) -> dict:
    """
    Execute the full evaluation pipeline.

    Parameters
    ----------
    scored_df : pd.DataFrame, optional
        Scored DataFrame. If None, read from the default output CSV.

    Returns
    -------
    dict
        Dictionary with keys: ks, gini, auc.
    """
    if scored_df is None:
        scored_df = pd.read_csv(config.SCORECARD_OUTPUT_CSV)

    y_true = scored_df[config.TARGET_COLUMN].values
    y_proba = scored_df["predicted_proba"].values
    credit_scores = scored_df["credit_score"].values

    # Compute metrics
    ks = plot_ks_curve(y_true, y_proba)
    auc = plot_roc_curve(y_true, y_proba)
    gini = compute_gini(auc)

    # Decile analysis
    dec_table = decile_table(y_true, credit_scores)
    plot_decile_chart(dec_table)

    # Print summary
    print_summary(ks, gini, auc, scored_df)

    return {"ks": ks, "gini": gini, "auc": auc}


if __name__ == "__main__":
    metrics = run()
    print("Evaluation complete.")
    print(f"  KS  = {metrics['ks']:.4f}")
    print(f"  Gini = {metrics['gini']:.4f}")
    print(f"  AUC  = {metrics['auc']:.4f}")
