-- ============================================================================
-- analysis_queries.sql
-- Credit Risk Scorecard – Analytical SQL Queries
--
-- Five production-ready queries for portfolio analysis and reporting.
-- Assumes tables created by create_tables.sql are populated.
-- ============================================================================


-- ─── 1. Default Rate by Income Band ─────────────────────────────────────────
-- Buckets annual_inc into five bands and shows the count of applications,
-- number of defaults, and default rate (%) for each band.

SELECT
    CASE
        WHEN annual_inc < 25000               THEN '< 25k'
        WHEN annual_inc BETWEEN 25000 AND 49999  THEN '25k - 50k'
        WHEN annual_inc BETWEEN 50000 AND 74999  THEN '50k - 75k'
        WHEN annual_inc BETWEEN 75000 AND 99999  THEN '75k - 100k'
        ELSE '100k+'
    END                                        AS income_band,
    COUNT(*)                                   AS total_loans,
    SUM(bad_loan)                              AS defaults,
    ROUND(AVG(bad_loan) * 100, 2)              AS default_rate_pct
FROM
    loan_applications
GROUP BY
    income_band
ORDER BY
    MIN(annual_inc);


-- ─── 2. Default Rate by Loan Purpose ────────────────────────────────────────
-- Shows default rate for every distinct loan purpose, ordered by volume.
-- NOTE: If 'purpose' was not retained in the cleaned schema, join with a
--       raw staging table or add the column to loan_applications.

SELECT
    purpose,
    COUNT(*)                                   AS total_loans,
    SUM(bad_loan)                              AS defaults,
    ROUND(AVG(bad_loan) * 100, 2)              AS default_rate_pct
FROM
    loan_applications
GROUP BY
    purpose
ORDER BY
    total_loans DESC;


-- ─── 3. Average Credit Score by Employment Length ───────────────────────────
-- Joins scored_results with loan_applications to show how creditworthiness
-- varies with employment tenure.

SELECT
    la.emp_length,
    COUNT(*)                                   AS total_loans,
    ROUND(AVG(sr.credit_score), 1)             AS avg_credit_score,
    ROUND(AVG(sr.predicted_proba) * 100, 2)    AS avg_default_prob_pct
FROM
    scored_results sr
    INNER JOIN loan_applications la ON sr.loan_id = la.loan_id
GROUP BY
    la.emp_length
ORDER BY
    avg_credit_score DESC;


-- ─── 4. Portfolio Concentration by Risk Tier ────────────────────────────────
-- Counts applications per risk tier and calculates each tier's share of the
-- total portfolio.

SELECT
    risk_tier,
    COUNT(*)                                   AS tier_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        2
    )                                          AS pct_of_total
FROM
    scored_results
GROUP BY
    risk_tier
ORDER BY
    MIN(credit_score) DESC;


-- ─── 5. Monthly Default Trend ───────────────────────────────────────────────
-- Groups by the loan issue month to reveal temporal trends in default rates.
-- Requires an 'issue_d' column (date/varchar) in loan_applications.
-- If issue_d is stored as 'Mon-YYYY' text, cast accordingly.

SELECT
    TO_CHAR(issue_d::DATE, 'YYYY-MM')          AS issue_month,
    COUNT(*)                                    AS total_loans,
    SUM(bad_loan)                               AS defaults,
    ROUND(AVG(bad_loan) * 100, 2)                AS default_rate_pct
FROM
    loan_applications
WHERE
    issue_d IS NOT NULL
GROUP BY
    issue_month
ORDER BY
    issue_month;
