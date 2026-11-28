# Member 6 Reproducibility

**Result: PASS**

A separate local clone with the current source overlay and an isolated Python environment was used. The external GitHub clone/download setup was not tested. All generated analyses remained in the separate checkout; Ivy’s original validation script and saved results in the main checkout were preserved.

Frozen dataset SHA-256: `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d`.

The environment was created and its pinned packages installed in an earlier attempt in this session, then reused for the successful pipeline run. The installation log and checksum are recorded in the audit. All 58 checks passed. The 28 CSV comparisons cover 27 regenerated tables and the preserved feature specification input.

## Checks

| Check | Passed | Detail |
| --- | :---: | --- |
| isolated_environment | Yes | Separate virtual environment |
| command_tests | Yes | exit 0; tests.log |
| command_cleaning | Yes | exit 0; cleaning.log |
| command_member1 | Yes | exit 0; member1.log |
| command_member2 | Yes | exit 0; member2.log |
| command_member3 | Yes | exit 0; member3.log |
| command_member4 | Yes | exit 0; member4.log |
| command_member5 | Yes | exit 0; member5.log |
| cleaned_csv_content | Yes | SHA-256 of decompressed CSV (gzip timestamps ignored) |
| cleaning_report | Yes | Exact rows, exclusions, duplicates, outcomes, dates, invalid values and missingness |
| table_data_dictionary.csv | Yes | maximum absolute numeric difference 0 |
| table_default_rate_by_annual_inc_bins.csv | Yes | maximum absolute numeric difference 0 |
| table_default_rate_by_fico_score_bins.csv | Yes | maximum absolute numeric difference 0 |
| table_default_rate_by_grade.csv | Yes | maximum absolute numeric difference 0 |
| table_default_rate_by_int_rate_bins.csv | Yes | maximum absolute numeric difference 0 |
| table_default_rate_by_issue_quarter.csv | Yes | maximum absolute numeric difference 0 |
| table_default_rate_by_purpose.csv | Yes | maximum absolute numeric difference 0 |
| table_eda_correlation_matrix.csv | Yes | maximum absolute numeric difference 3.89e-16 |
| table_eda_numeric_summary.csv | Yes | maximum absolute numeric difference 0 |
| table_feature_specification.csv | Yes | maximum absolute numeric difference 0 |
| table_inference_global_tests.csv | Yes | maximum absolute numeric difference 5.19e-09 |
| table_inference_odds_ratios.csv | Yes | maximum absolute numeric difference 7.45e-11 |
| table_inference_sample_summary.csv | Yes | maximum absolute numeric difference 1.33e-14 |
| table_logistic_interpretation.csv | Yes | maximum absolute numeric difference 0 |
| table_missingness.csv | Yes | maximum absolute numeric difference 0 |
| table_model_comparison.csv | Yes | maximum absolute numeric difference 0 |
| table_model_specifications.csv | Yes | maximum absolute numeric difference 0 |
| table_out_of_time_predictions.csv | Yes | maximum absolute numeric difference 2.78e-16 |
| table_temporal_split.csv | Yes | maximum absolute numeric difference 0 |
| table_validation_calibration_bins.csv | Yes | maximum absolute numeric difference 0 |
| table_validation_missingness_by_partition.csv | Yes | maximum absolute numeric difference 0 |
| table_validation_overall_metrics.csv | Yes | maximum absolute numeric difference 8.88e-16 |
| table_validation_paired_model_deltas.csv | Yes | maximum absolute numeric difference 0 |
| table_validation_robustness_checks.csv | Yes | maximum absolute numeric difference 2e-15 |
| table_validation_subgroup_performance.csv | Yes | maximum absolute numeric difference 1.11e-15 |
| table_validation_threshold_curve_validation.csv | Yes | maximum absolute numeric difference 0 |
| table_validation_threshold_performance.csv | Yes | maximum absolute numeric difference 0 |
| table_validation_vintage_default_rates.csv | Yes | maximum absolute numeric difference 0 |
| manifest_outputs/figures/validation_calibration_plot.png | Yes | Generated output matches Ivy manifest |
| manifest_outputs/figures/validation_roc_pr_curves.png | Yes | Generated output matches Ivy manifest |
| manifest_outputs/figures/validation_threshold_tradeoffs.png | Yes | Generated output matches Ivy manifest |
| manifest_outputs/tables/validation_calibration_bins.csv | Yes | Generated output matches Ivy manifest |
| manifest_outputs/tables/validation_missingness_by_partition.csv | Yes | Generated output matches Ivy manifest |
| manifest_outputs/tables/validation_overall_metrics.csv | Yes | Generated output matches Ivy manifest |
| manifest_outputs/tables/validation_paired_model_deltas.csv | Yes | Generated output matches Ivy manifest |
| manifest_outputs/tables/validation_robustness_checks.csv | Yes | Generated output matches Ivy manifest |
| manifest_outputs/tables/validation_subgroup_performance.csv | Yes | Generated output matches Ivy manifest |
| manifest_outputs/tables/validation_threshold_curve_validation.csv | Yes | Generated output matches Ivy manifest |
| manifest_outputs/tables/validation_threshold_performance.csv | Yes | Generated output matches Ivy manifest |
| manifest_outputs/tables/validation_vintage_default_rates.csv | Yes | Generated output matches Ivy manifest |
| manifest_report/member5_validation_report.md | Yes | Generated output matches Ivy manifest |
| robustness_no_suppressed_failures | Yes | 0 failed robustness rows |
| repeat_eda_command | Yes | Same input and environment |
| repeat_eda_outputs | Yes | Changed files: [] |
| no_tracked_data_or_models | Yes | [] |
| predictor_leakage_contract | Yes | Ivy contract rejects outcome/audit and post-origination fields |
| training_only_preprocessing_recorded | Yes | Minimum validation log loss within each family, then across families |
| raw_data_unchanged | Yes | Frozen raw checksum before/after |

## Commands actually run

```text
git clone --no-hardlinks --quiet <project> <workdir>/checkout  # exit=0, seconds=0.07
<reused-environment>/venv/bin/python -m pytest -q  # exit=0, seconds=14.42
<reused-environment>/venv/bin/python src/lending_club_data_cleaning.py --input data/raw/accepted_2007_to_2018Q4.csv.gz --output-dir data/cleaned/clean_v1  # exit=0, seconds=62.29
<reused-environment>/venv/bin/python analysis/01_cleaning_member2.py  # exit=0, seconds=25.87
<reused-environment>/venv/bin/python analysis/02_visualization_member3.py  # exit=0, seconds=4.69
<reused-environment>/venv/bin/python analysis/03_inference_member4.py  # exit=0, seconds=69.48
<reused-environment>/venv/bin/python analysis/04_modelling_pavel.py  # exit=0, seconds=215.89
<reused-environment>/venv/bin/python analysis/05_validation_member5.py  # exit=0, seconds=410.38
<reused-environment>/venv/bin/python analysis/02_visualization_member3.py  # exit=0, seconds=4.6
```

## Environment and comparison rules

Integer counts, column order, row order and categorical values must match exactly. Numeric CSV results use relative tolerance 1e-6 and absolute tolerance 1e-10. The cleaned CSV is compared after decompression. EDA repeat outputs are compared byte-for-byte. Bootstrap uncertainty describes sampling variation conditional on the fitted models; package/plot rendering differences can prevent binary agreement across environments.

Ivy's optional total-issued-loan denominator was not supplied, so her report omits the share of the full 2018 vintage retained after outcome exclusion; this does not affect the reported resolved-cohort metrics.

The fresh run is compared with earlier saved results. Only EDA receives an additional immediate repeat; a second full pipeline execution is not claimed. Automated contract checks and synthetic tests support the leakage review; the metadata alone does not prove absence of leakage.

This is a member reproducibility note, not the final group report. Code and execution were assisted by Codex. Course approach: Week 1 reproducible seeds; Week 8 seeded simulation/bootstrap; Week 9 training-only preprocessing and validation/test separation.

```text
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
fonttools==4.68.0
formulaic==1.2.2
iniconfig==2.3.1
interface_meta==2.0.1
joblib==1.6.0
kiwisolver==1.5.1
matplotlib==3.11.2
narwhals==2.28.0
numpy==2.5.4
packaging==26.3
pandas==2.3.3
pandas-stubs==2.3.3.260113
patsy==1.0.3
pillow==13.0.0
pluggy==1.7.0
Pygments==2.21.0
pyparsing==3.3.3
pytest==8.4.2
python-dateutil==4.0.0
pytz==2026.6
scikit-learn==1.9.1
scipy==1.18.1
seaborn==0.13.2
statsmodels==0.15.0
threadpoolctl==3.8.0
types-pytz==2026.6.0.20261031
typing_extensions==4.16.0
tzdata==2026.6
wrapt==2.5.2
```

Detailed audit: `outputs/tables/reproducibility_audit.json`. Per-command logs remain in the local work directory recorded there.
