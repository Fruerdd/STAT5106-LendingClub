# Member 3 — Exploratory Data Analysis Report

**Project:** STAT5106 Lending Club Loan Default  
**Analysis:** Visualization and Exploratory Data Analysis  
**Purpose:** Provide descriptive findings and data-quality considerations for subsequent statistical inference and predictive modelling.

## 1. Dataset and Approach

The analysis uses the cleaned Lending Club dataset, restricted to loans with resolved outcomes: Fully Paid (default = 0) and Charged Off or Default (default = 1).

The exploratory analysis examines distributions of loan and borrower characteristics, observed default rates across borrower groups, temporal patterns, variable correlations, and missingness.

All findings are descriptive and should not be interpreted as causal relationships or evidence of independent statistical significance.

## 2. Key Findings

### 2.1 Loan Grade and Interest Rate

Observed default rates increase substantially across loan grades, from approximately 6% for Grade A to nearly 50% for Grade G.

Similarly, default rates increase across interest-rate categories, suggesting that the interest rate assigned at origination reflects differences in borrower risk.

### 2.2 Borrower Creditworthiness

FICO score exhibits an inverse relationship with observed default. Borrowers with higher origination FICO scores generally show lower default rates.

Annual income also exhibits a negative descriptive association with default, although the relationship may be influenced by other borrower characteristics.

### 2.3 Loan Purpose

Observed default rates differ across loan purposes. Category-specific sample sizes should be considered before interpreting differences, particularly for less common purposes.

### 2.4 Temporal Trends

Default rates vary across loan origination quarters. However, the restriction to resolved outcomes may lead to unequal follow-up and selection effects. Temporal patterns should therefore be interpreted cautiously.

## 3. Data Quality Considerations

Missingness is particularly substantial in certain credit-history variables. Missing values may sometimes represent the absence of a recorded credit event rather than a conventional data collection failure.

The correlation analysis also reveals relationships among selected borrower and loan characteristics, including FICO score, interest rate, revolving utilization, and loan term.

These correlations should be considered during subsequent feature engineering and model specification.

## 4. Recommendations for the next members

The following variables are promising candidates for further investigation:

- Loan grade
- Origination FICO score
- Interest rate
- Annual income
- Loan purpose
- Loan term and debt-to-income ratio

Before modelling, the next team members should:

- Examine subgroup sample sizes and sparse categories.
- Investigate missingness mechanisms and appropriate imputation.
- Evaluate correlations among credit-risk variables.
- Preserve the out-of-time evaluation design.
- Avoid post-outcome predictors and held-out data in feature selection.

Please note, the EDA does not establish which variables are independently significant or which feature combination yields the best predictive performance.

## 5. Reproducibility

The analysis can be reproduced from the project root after generating the cleaned dataset:

```bash
python analysis/01_cleaning_member2.py --input data/cleaned/clean_v1/lending_club_clean.csv.gz

python analysis/02_visualization_member3.py --input data/cleaned/clean_v1/lending_club_clean.csv.gz
```

Figures are saved under `outputs/figures/`, and summary statistics are saved under `outputs/tables/`.

All reported results should be checked against the generated CSV tables before final submission.