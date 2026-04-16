"""
generate_powerbi_data.py – Auto-generate Power BI-ready summary datasets.

This script reads the scored output from the pipeline and produces multiple
pre-aggregated CSV files optimized for Power BI import. All KPIs, measures,
and calculated columns are pre-computed so that building the dashboard requires
only drag-and-drop — NO DAX formulas or Power Query transformations needed.

Output files (saved to dashboard/powerbi_data/):
  1. kpi_summary.csv           – Single-row KPI card values
  2. risk_tier_summary.csv     – Risk tier breakdown with counts, rates, avg scores
  3. score_distribution.csv    – Binned score histogram data (20-pt bins)
  4. decile_analysis.csv       – Decile-level bad rate and cumulative capture
  5. iv_feature_importance.csv – Feature importance by Information Value
  6. grade_analysis.csv        – Default rate and avg score by loan grade
  7. home_ownership_analysis.csv – Breakdown by home ownership status
  8. purpose_analysis.csv      – Default rate by loan purpose
  9. monthly_trend.csv         – Monthly default rate trend (if issue_d available)
  10. scored_sample.csv        – 50K random sample of scored loans for detail views

Usage:
    python dashboard/generate_powerbi_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Add project root to path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ─── Output directory ─────────────────────────────────────────────────────────
POWERBI_DATA_DIR = Path(__file__).resolve().parent / "powerbi_data"
POWERBI_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple:
    """Load the scored output and IV summary."""
    print("Loading scored output...")
    scored = pd.read_csv(config.SCORECARD_OUTPUT_CSV, low_memory=False)
    print(f"  Loaded {len(scored):,} rows")

    iv_df = None
    if config.IV_SUMMARY_CSV.exists():
        iv_df = pd.read_csv(config.IV_SUMMARY_CSV)
    return scored, iv_df


def add_score_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add a human-readable score band column."""
    bins = [299, 599, 659, 719, 799, 901]
    labels = ["300-599", "600-659", "660-719", "720-799", "800-900"]
    df["score_band"] = pd.cut(df["credit_score"], bins=bins, labels=labels)
    return df


def generate_kpi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a single-row KPI summary for card visuals."""
    total_loans = len(df)
    total_defaults = df["bad_loan"].sum()
    default_rate = total_defaults / total_loans * 100
    avg_score = df["credit_score"].mean()
    median_score = df["credit_score"].median()
    approval_rate = (df["credit_score"] >= config.APPROVAL_CUTOFF).mean() * 100
    avg_loan_amount = df["loan_amnt"].mean() if "loan_amnt" in df.columns else 0
    avg_interest_rate = df["int_rate"].mean() if "int_rate" in df.columns else 0

    kpi = pd.DataFrame([{
        "Total Loans": total_loans,
        "Total Defaults": int(total_defaults),
        "Default Rate (%)": round(default_rate, 2),
        "Average Credit Score": round(avg_score, 1),
        "Median Credit Score": round(median_score, 1),
        "Approval Rate (%)": round(approval_rate, 1),
        "Average Loan Amount ($)": round(avg_loan_amount, 0),
        "Average Interest Rate (%)": round(avg_interest_rate, 2),
    }])
    return kpi


def generate_risk_tier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate risk tier breakdown table."""
    tier_order = ["Super Prime", "Prime", "Near Prime", "Subprime", "Deep Subprime"]

    summary = df.groupby("risk_tier").agg(
        Count=("bad_loan", "size"),
        Defaults=("bad_loan", "sum"),
        Avg_Credit_Score=("credit_score", "mean"),
        Min_Score=("credit_score", "min"),
        Max_Score=("credit_score", "max"),
        Avg_Loan_Amount=("loan_amnt", "mean") if "loan_amnt" in df.columns else ("bad_loan", "size"),
    ).reset_index()

    summary["Default Rate (%)"] = (summary["Defaults"] / summary["Count"] * 100).round(2)
    summary["Portfolio Share (%)"] = (summary["Count"] / summary["Count"].sum() * 100).round(1)
    summary["Avg_Credit_Score"] = summary["Avg_Credit_Score"].round(0)

    # Sort in tier order
    summary["risk_tier"] = pd.Categorical(summary["risk_tier"], categories=tier_order, ordered=True)
    summary = summary.sort_values("risk_tier").reset_index(drop=True)

    summary.columns = [
        "Risk Tier", "Count", "Defaults", "Avg Credit Score",
        "Min Score", "Max Score", "Avg Loan Amount",
        "Default Rate (%)", "Portfolio Share (%)"
    ]
    return summary


def generate_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Generate score histogram data with 20-point bins."""
    bins = list(range(300, 920, 20))
    labels = [f"{b}-{b+19}" for b in bins[:-1]]
    df["score_bin"] = pd.cut(df["credit_score"], bins=bins, labels=labels, right=False)

    dist = df.groupby("score_bin", observed=True).agg(
        Count=("bad_loan", "size"),
        Defaults=("bad_loan", "sum"),
    ).reset_index()

    dist["Default Rate (%)"] = (dist["Defaults"] / dist["Count"] * 100).round(2)
    dist.columns = ["Score Range", "Count", "Defaults", "Default Rate (%)"]
    return dist


def generate_decile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Generate decile analysis table."""
    tmp = df[["credit_score", "bad_loan"]].copy()
    tmp = tmp.sort_values("credit_score", ascending=False).reset_index(drop=True)
    tmp["Decile"] = pd.qcut(tmp.index, 10, labels=False, duplicates="drop") + 1

    agg = tmp.groupby("Decile").agg(
        Count=("bad_loan", "size"),
        Defaults=("bad_loan", "sum"),
        Min_Score=("credit_score", "min"),
        Max_Score=("credit_score", "max"),
        Avg_Score=("credit_score", "mean"),
    ).reset_index()

    agg["Bad Rate (%)"] = (agg["Defaults"] / agg["Count"] * 100).round(2)
    agg["Cumulative Defaults"] = agg["Defaults"].cumsum()
    agg["Cumulative Bad Capture (%)"] = (agg["Cumulative Defaults"] / agg["Defaults"].sum() * 100).round(2)
    agg["Avg_Score"] = agg["Avg_Score"].round(0)

    agg.columns = [
        "Decile", "Count", "Defaults", "Min Score", "Max Score", "Avg Score",
        "Bad Rate (%)", "Cumulative Defaults", "Cumulative Bad Capture (%)"
    ]
    return agg


def generate_grade_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Generate analysis by loan grade (A-G)."""
    if "grade" not in df.columns:
        return pd.DataFrame()

    grade = df.groupby("grade").agg(
        Count=("bad_loan", "size"),
        Defaults=("bad_loan", "sum"),
        Avg_Credit_Score=("credit_score", "mean"),
        Avg_Interest_Rate=("int_rate", "mean") if "int_rate" in df.columns else ("bad_loan", "size"),
        Avg_Loan_Amount=("loan_amnt", "mean") if "loan_amnt" in df.columns else ("bad_loan", "size"),
    ).reset_index()

    grade["Default Rate (%)"] = (grade["Defaults"] / grade["Count"] * 100).round(2)
    grade["Avg_Credit_Score"] = grade["Avg_Credit_Score"].round(0)
    grade = grade.sort_values("grade")

    grade.columns = [
        "Grade", "Count", "Defaults", "Avg Credit Score",
        "Avg Interest Rate (%)", "Avg Loan Amount ($)", "Default Rate (%)"
    ]
    return grade


def generate_home_ownership_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Generate analysis by home ownership status."""
    if "home_ownership" not in df.columns:
        return pd.DataFrame()

    ho = df.groupby("home_ownership").agg(
        Count=("bad_loan", "size"),
        Defaults=("bad_loan", "sum"),
        Avg_Credit_Score=("credit_score", "mean"),
    ).reset_index()

    ho["Default Rate (%)"] = (ho["Defaults"] / ho["Count"] * 100).round(2)
    ho["Portfolio Share (%)"] = (ho["Count"] / ho["Count"].sum() * 100).round(1)
    ho["Avg_Credit_Score"] = ho["Avg_Credit_Score"].round(0)
    ho = ho.sort_values("Count", ascending=False)

    ho.columns = [
        "Home Ownership", "Count", "Defaults", "Avg Credit Score",
        "Default Rate (%)", "Portfolio Share (%)"
    ]
    return ho


def generate_scored_sample(df: pd.DataFrame, n: int = 50000) -> pd.DataFrame:
    """Generate a random sample of scored loans for detail-level visuals."""
    sample = df.sample(n=min(n, len(df)), random_state=42).copy()

    # Keep only columns useful for Power BI
    keep_cols = [c for c in [
        "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
        "grade", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "dti", "delinq_2yrs", "revol_bal",
        "revol_util", "total_acc", "bad_loan",
        "predicted_proba", "credit_score", "risk_tier",
    ] if c in sample.columns]

    sample = sample[keep_cols].reset_index(drop=True)
    sample = add_score_band(sample)
    return sample


def main():
    """Generate all Power BI-ready datasets."""
    print("=" * 60)
    print("  GENERATING POWER BI-READY DATASETS")
    print("=" * 60)

    scored_df, iv_df = load_data()
    scored_df = add_score_band(scored_df)

    # 1. KPI Summary
    print("\n[1/8] KPI Summary...")
    kpi = generate_kpi_summary(scored_df)
    kpi.to_csv(POWERBI_DATA_DIR / "kpi_summary.csv", index=False)
    print(f"  Saved: kpi_summary.csv")
    print(f"  Default Rate: {kpi['Default Rate (%)'].values[0]:.2f}%")
    print(f"  Avg Score: {kpi['Average Credit Score'].values[0]:.1f}")

    # 2. Risk Tier Summary
    print("\n[2/8] Risk Tier Summary...")
    tiers = generate_risk_tier_summary(scored_df)
    tiers.to_csv(POWERBI_DATA_DIR / "risk_tier_summary.csv", index=False)
    print(f"  Saved: risk_tier_summary.csv ({len(tiers)} tiers)")

    # 3. Score Distribution
    print("\n[3/8] Score Distribution...")
    dist = generate_score_distribution(scored_df)
    dist.to_csv(POWERBI_DATA_DIR / "score_distribution.csv", index=False)
    print(f"  Saved: score_distribution.csv ({len(dist)} bins)")

    # 4. Decile Analysis
    print("\n[4/8] Decile Analysis...")
    decile = generate_decile_analysis(scored_df)
    decile.to_csv(POWERBI_DATA_DIR / "decile_analysis.csv", index=False)
    print(f"  Saved: decile_analysis.csv ({len(decile)} deciles)")

    # 5. Feature Importance (IV)
    print("\n[5/8] Feature Importance (IV)...")
    if iv_df is not None:
        iv_df.columns = ["Feature", "Information Value"]
        iv_df = iv_df.sort_values("Information Value", ascending=False)
        iv_df.to_csv(POWERBI_DATA_DIR / "iv_feature_importance.csv", index=False)
        print(f"  Saved: iv_feature_importance.csv ({len(iv_df)} features)")
    else:
        print("  Skipped (iv_summary.csv not found)")

    # 6. Grade Analysis
    print("\n[6/8] Grade Analysis...")
    grade = generate_grade_analysis(scored_df)
    if not grade.empty:
        grade.to_csv(POWERBI_DATA_DIR / "grade_analysis.csv", index=False)
        print(f"  Saved: grade_analysis.csv ({len(grade)} grades)")
    else:
        print("  Skipped (grade column not found)")

    # 7. Home Ownership Analysis
    print("\n[7/8] Home Ownership Analysis...")
    ho = generate_home_ownership_analysis(scored_df)
    if not ho.empty:
        ho.to_csv(POWERBI_DATA_DIR / "home_ownership_analysis.csv", index=False)
        print(f"  Saved: home_ownership_analysis.csv ({len(ho)} categories)")
    else:
        print("  Skipped (home_ownership column not found)")

    # 8. Scored Sample (50K rows for detail views)
    print("\n[8/8] Scored Sample (50K)...")
    sample = generate_scored_sample(scored_df)
    sample.to_csv(POWERBI_DATA_DIR / "scored_sample.csv", index=False)
    print(f"  Saved: scored_sample.csv ({len(sample):,} rows)")

    # Summary
    files = list(POWERBI_DATA_DIR.glob("*.csv"))
    print("\n" + "=" * 60)
    print(f"  DONE — {len(files)} files saved to:")
    print(f"  {POWERBI_DATA_DIR}")
    print("=" * 60)
    print("\n  NEXT STEPS:")
    print("  1. Open Power BI Desktop")
    print("  2. Get Data -> Text/CSV -> select ALL files in dashboard/powerbi_data/")
    print("  3. Each CSV becomes a separate table -- just drag columns onto visuals")
    print("  4. No DAX needed -- all measures are pre-computed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
