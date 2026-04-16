-- ============================================================================
-- create_tables.sql
-- Credit Risk Scorecard – Database Schema
--
-- Creates the core tables for storing loan application data and scored results.
-- Compatible with PostgreSQL (use psycopg2 + SQLAlchemy for Python integration).
-- ============================================================================

-- ─── Loan Applications Table ────────────────────────────────────────────────
-- Stores the cleaned loan application records after data loading and feature
-- engineering.  Mirrors the columns produced by data_loader.py.

CREATE TABLE IF NOT EXISTS loan_applications (
    loan_id             SERIAL PRIMARY KEY,
    loan_amnt           NUMERIC(12, 2),
    funded_amnt         NUMERIC(12, 2),
    term                INTEGER,
    int_rate            NUMERIC(6, 2),
    installment         NUMERIC(10, 2),
    grade               VARCHAR(2),
    emp_length          VARCHAR(20),
    home_ownership      VARCHAR(20),
    annual_inc          NUMERIC(14, 2),
    verification_status VARCHAR(30),
    loan_status         VARCHAR(20),
    dti                 NUMERIC(8, 2),
    delinq_2yrs         INTEGER,
    fico_range_low      INTEGER,
    fico_range_high     INTEGER,
    open_acc            INTEGER,
    pub_rec             INTEGER,
    revol_bal           NUMERIC(14, 2),
    revol_util          NUMERIC(6, 2),
    total_acc           INTEGER,
    out_prncp           NUMERIC(14, 2),
    total_pymnt         NUMERIC(14, 2),
    last_pymnt_amnt     NUMERIC(14, 2),
    bad_loan            SMALLINT NOT NULL DEFAULT 0,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE  loan_applications IS 'Cleaned Lending Club loan records (Fully Paid / Charged Off only).';
COMMENT ON COLUMN loan_applications.bad_loan IS 'Binary target – 1 = Charged Off (default), 0 = Fully Paid.';


-- ─── Scored Results Table ───────────────────────────────────────────────────
-- Stores the output of the scorecard model after scoring each application.

CREATE TABLE IF NOT EXISTS scored_results (
    score_id        SERIAL PRIMARY KEY,
    loan_id         INTEGER NOT NULL REFERENCES loan_applications(loan_id),
    credit_score    INTEGER NOT NULL,
    risk_tier       VARCHAR(20) NOT NULL,
    predicted_proba NUMERIC(8, 6) NOT NULL,
    scored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE  scored_results IS 'Credit scores and risk tiers assigned by the scorecard model.';
COMMENT ON COLUMN scored_results.credit_score IS 'Scaled credit score in the 300–900 range.';
COMMENT ON COLUMN scored_results.risk_tier IS 'Risk category: Super Prime / Prime / Near Prime / Subprime / Deep Subprime.';


-- ─── Indexes ────────────────────────────────────────────────────────────────
-- Speed up common look-ups and aggregation queries.

CREATE INDEX IF NOT EXISTS idx_scored_risk_tier
    ON scored_results (risk_tier);

CREATE INDEX IF NOT EXISTS idx_scored_credit_score
    ON scored_results (credit_score);

CREATE INDEX IF NOT EXISTS idx_loan_app_bad_loan
    ON loan_applications (bad_loan);

CREATE INDEX IF NOT EXISTS idx_loan_app_grade
    ON loan_applications (grade);
