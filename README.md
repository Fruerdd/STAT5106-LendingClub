# STAT5106 Lending Club Loan Default

A local Python/PyCharm group project for predicting Lending Club loan default.

**Research question:** Which borrower and loan characteristics significantly predict loan
default, and how accurately can default be predicted?

The cleaning pipeline restricts the main cohort to resolved outcomes: Fully Paid (`0`) and
Charged Off or Default (`1`). Current, late, grace-period, missing, and unknown outcomes are
excluded. Only application/origination-time fields are read; post-outcome fields are excluded
to reduce leakage. Statistical significance and predictive accuracy must be assessed separately.

## Prerequisites

- Python 3.12 recommended (a 64-bit Python 3.10+ environment should also work).
- PyCharm for local development; Google Colab is not used.
- Git and, for maintainers, the authenticated GitHub CLI (`gh`).
- The frozen course dataset, supplied separately rather than through GitHub.

## PyCharm and virtual-environment setup

1. Open the `STAT5106-LendingClub` directory as a PyCharm project.
2. Create a local interpreter if `.venv` does not already exist:

   macOS/Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   Windows (Command Prompt; each line can also be run independently in PowerShell):

   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   One-line Windows PowerShell dependency setup after creating `.venv`:

   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. In PyCharm, open **Settings/Preferences → Project → Python Interpreter**, then select the
   existing interpreter at `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe`
   (Windows).
4. Set run-configuration working directories to the repository root. Do not store personal
   absolute paths in source files or run configurations that will be shared.

## Exact local dataset placement

Put the frozen file manually at:

```text
data/raw/accepted_2007_to_2018Q4.csv.gz
```

Use the same dataset version on every machine. The raw file must never be edited, committed,
or pushed. The raw and cleaned data directories are ignored even though the GitHub repository
is private; this keeps large/sensitive source and derived data out of version control. Small
CSV result tables under `outputs/tables` remain trackable because there is deliberately no
global `*.csv` ignore rule. See `data/raw/README.md` and `CONTRIBUTING.md`.

## Install, clean, and test

Run all commands from the repository root. Install dependencies as above, then run:

```bash
python src/lending_club_data_cleaning.py --input data/raw/accepted_2007_to_2018Q4.csv.gz --output-dir data/cleaned/clean_v1
```

This reads the gzip file directly with `usecols`, preserves the original `loan_status`, and
writes:

- `data/cleaned/clean_v1/lending_club_clean.csv.gz`
- `data/cleaned/clean_v1/cleaning_report.json`

The report records row reconciliation, status exclusions, duplicates, invalid values,
missingness, outcomes, default rate, and issue-date range. Impossible values become missing;
legitimate extreme observations are retained. No full-dataset median imputation occurs.
Modelling imputers must be fitted on training data only.

Run the synthetic tests (no real dataset is needed):

```bash
python -m pytest
```

## Analysis execution order

The six entry points are initially scaffolds unless their docstrings state otherwise. They fail
clearly when required real inputs are missing and do not generate placeholder findings.

1. **Member 2 — cleaning validation, data dictionary, missingness and leakage audit**

   ```bash
   python analysis/01_cleaning_member2.py --input data/cleaned/clean_v1/lending_club_clean.csv.gz
   ```

2. **Member 3 — visualization and exploratory analysis**

   ```bash
   python analysis/02_visualization_member3.py --input data/cleaned/clean_v1/lending_club_clean.csv.gz
   ```

3. **Member 4 — statistical inference and feature engineering**

   ```bash
   python analysis/03_inference_member4.py --input data/cleaned/clean_v1/lending_club_clean.csv.gz
   ```

4. **Pavel — temporal split, train-only preprocessing, logistic regression, comparison and interpretation**

   ```bash
   python analysis/04_modelling_pavel.py --input data/cleaned/clean_v1/lending_club_clean.csv.gz
   ```

5. **Member 5 — out-of-time validation, robustness and limitations**

   ```bash
   python analysis/05_validation_member5.py --cleaned data/cleaned/clean_v1/lending_club_clean.csv.gz --predictions outputs/tables/out_of_time_predictions.csv
   ```

   The predictions table must be real model output with `id`, `issue_date`, `default`, and
   `predicted_probability`. It is not included and must be produced before validation.
6. **Member 6 — reproducibility and consistency checks**

   ```bash
   python analysis/06_reproducibility_member6.py --raw data/raw/accepted_2007_to_2018Q4.csv.gz --cleaned data/cleaned/clean_v1/lending_club_clean.csv.gz --report data/cleaned/clean_v1/cleaning_report.json
   ```

## Ownership and dependencies

| Owner | Responsibility | Depends on |
| --- | --- | --- |
| Member 2 | Cleaning validation, dictionary, missingness/leakage audits | Frozen raw data; cleaning module |
| Member 3 | Exploratory tables and visualizations | Cleaned cohort |
| Member 4 | Feature engineering, estimands, inference and uncertainty | Cleaned cohort; agreed features |
| Pavel | Temporal split, pipelines, logistic models, comparison and interpretation | Cleaned cohort; feature decisions |
| Member 5 | ROC-AUC, PR-AUC, calibration, thresholds, bootstrap, subgroup and robustness checks | Real out-of-time predictions and cohort |
| Member 6 | Fresh-clone reproduction and consistency checks | All preceding outputs and frozen inputs |

Changes to outcomes, features, splits, or the dataset version affect downstream owners and must
be documented and reviewed rather than made silently.

## Directory structure

```text
STAT5106-LendingClub/
├── data/
│   ├── raw/                 # Local frozen CSV.GZ; ignored
│   └── cleaned/             # Generated local cohorts/reports; ignored
├── src/                     # Shared paths and cleaning implementation
├── analysis/                # Ordered team entry points
├── tests/                   # Synthetic, dataset-independent tests
├── outputs/
│   ├── figures/             # Real generated figures
│   └── tables/              # Small real result/prediction tables
├── report/ai_use_log.md     # Functional AI-use disclosure
├── slides/                  # Presentation materials
├── .github/workflows/       # Dataset-independent CI tests
├── requirements.txt
├── README.md
└── CONTRIBUTING.md
```

`src/paths.py` derives all common paths from `__file__`; scripts can therefore run from any
checkout location without personal absolute paths. Cleaning and analysis commands are intended
to be launched from the repository root.

## Fresh-clone reproducibility procedure

1. Clone the private repository and enter its root:

   ```bash
   git clone <PRIVATE_REPOSITORY_URL>
   cd STAT5106-LendingClub
   ```

2. Create and select `.venv`, then install `requirements.txt` as described above.
3. Manually copy the team's same frozen dataset to
   `data/raw/accepted_2007_to_2018Q4.csv.gz`; it will not be downloaded by Git or CI.
4. Run `python -m pytest` and require all synthetic tests to pass.
5. Run the exact cleaning command above. Compare `cleaning_report.json` with the team's reviewed
   baseline, including row counts, exclusions, outcome counts, default rate, and date range.
6. Run analysis entry points in order. Resolve any missing real input instead of substituting
   synthetic results. Member 5 additionally needs the real out-of-time prediction file.
7. Run Member 6's consistency command and record any environment, dataset, feature, or output
   discrepancies. Do not commit generated data; use `git status --short` and
   `git check-ignore -v data/raw/accepted_2007_to_2018Q4.csv.gz` to verify local-only storage.

The synthetic tests and GitHub Actions do not require, access, or download the private dataset.
A real-data run must still be performed and checked by the team before results are reported.
