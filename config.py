"""
Configuration file for Credit Risk Scorecard project.
All thresholds, paths, hyperparameters, and constants are defined here.
"""

from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ─── Data Paths ──────────────────────────────────────────────────────────────
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
PLOTS_PATH = OUTPUTS_PATH / "plots"

# ─── Raw Data File ───────────────────────────────────────────────────────────
RAW_CSV = DATA_RAW_PATH / "loan.csv"

# ─── Processed Data Files ────────────────────────────────────────────────────
CLEANED_CSV = DATA_PROCESSED_PATH / "cleaned_loans.csv"
FEATURES_CSV = DATA_PROCESSED_PATH / "features.csv"
WOE_FEATURES_CSV = DATA_PROCESSED_PATH / "woe_features.csv"

# ─── Output Files ────────────────────────────────────────────────────────────
SCORECARD_OUTPUT_CSV = OUTPUTS_PATH / "scorecard_output.csv"
IV_SUMMARY_CSV = OUTPUTS_PATH / "iv_summary.csv"

# ─── Feature Selection ───────────────────────────────────────────────────────
IV_THRESHOLD = 0.10  # Minimum Information Value to keep a feature

# ─── Columns of Interest ────────────────────────────────────────────────────
TARGET_COLUMN = "bad_loan"
KEEP_LOAN_STATUS = ["Fully Paid", "Charged Off"]

KEY_COLUMNS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "loan_status", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "out_prncp",
    "total_pymnt", "last_pymnt_amnt",
]

# ─── Missing Value Threshold ────────────────────────────────────────────────
MISSING_THRESHOLD = 0.40  # Drop columns with >40% missing values

# ─── Model Hyperparameters ───────────────────────────────────────────────────
TEST_SIZE = 0.30
RANDOM_STATE = 42

# ─── Scorecard PDO Scaling ───────────────────────────────────────────────────
BASE_SCORE = 600
PDO = 20            # Points to Double Odds
BASE_ODDS = 1 / 19  # Roughly 5% default rate baseline
SCORE_MIN = 300
SCORE_MAX = 900

# ─── Risk Tier Definitions ──────────────────────────────────────────────────
RISK_TIERS = {
    "Super Prime":    (800, 900),
    "Prime":          (720, 799),
    "Near Prime":     (660, 719),
    "Subprime":       (600, 659),
    "Deep Subprime":  (300, 599),
}

# ─── Plot Settings ───────────────────────────────────────────────────────────
PLOT_DPI = 150
APPROVAL_CUTOFF = 660  # Score threshold for approval rate computation

# ─── Logging Format ──────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
