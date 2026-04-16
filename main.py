"""
main.py – Single entry point for the Credit Risk Scorecard pipeline.

Runs all five pipeline steps in sequence:
  1. Data Loading & Cleaning
  2. Feature Engineering
  3. WoE / IV Computation
  4. Scorecard Model Training & Scoring
  5. Model Evaluation & Diagnostics

Usage:
    python main.py
"""

import time
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data_loader, feature_engineering, woe_iv, scorecard, evaluation  # noqa: E402
import config  # noqa: E402


def main() -> None:
    """Run the full credit risk scorecard pipeline end to end."""
    start = time.time()

    # Ensure output directories exist
    config.DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    config.PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data Loading ─────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1 / 5 — Data Loading & Cleaning")
    print("=" * 60)
    try:
        cleaned_df = data_loader.run()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("Aborting pipeline.")
        sys.exit(1)
    print(f"  Data loaded: {len(cleaned_df)} rows")

    # ── Step 2: Feature Engineering ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 / 5 — Feature Engineering")
    print("=" * 60)
    features_df = feature_engineering.run()
    print(f"  Features engineered: {features_df.shape[1]} columns")

    # ── Step 3: WoE / IV ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 / 5 — WoE Encoding & IV Filtering")
    print("=" * 60)
    woe_df, iv_table = woe_iv.run()
    print("\n  Information Value Summary:")
    print(iv_table.to_string(index=False))
    print(f"\n  WoE features retained: {woe_df.shape[1] - 1}")

    # ── Step 4: Scorecard Model ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 / 5 — Scorecard Model & Scoring")
    print("=" * 60)
    scored_df, model, X_test, y_test = scorecard.run()
    print(f"  Scored {len(scored_df)} applications")
    print(f"  Score range: {scored_df['credit_score'].min()} – {scored_df['credit_score'].max()}")
    print(f"\n  Score distribution:")
    print(scored_df["risk_tier"].value_counts().to_string())

    # ── Step 5: Evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 / 5 — Model Evaluation")
    print("=" * 60)
    metrics = evaluation.run(scored_df)

    # ── Finish ──────────────────────────────────────────────────────────
    elapsed = time.time() - start
    minutes, seconds = divmod(elapsed, 60)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total runtime : {int(minutes)}m {seconds:.1f}s")
    print(f"  Outputs saved to : {config.OUTPUTS_PATH}")
    print(f"  Plots saved to   : {config.PLOTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
