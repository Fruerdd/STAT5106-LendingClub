# Lending Club research dashboard

An interactive display of the existing STAT5106 analysis, including Ivy's original validation results. The dashboard uses aggregate tables and downsampled ROC/PR curves. It contains no individual borrower records or loan-level predictions.

Open `dist/index.html` directly for an offline view, or run this from the main repository:

```bash
.venv/bin/python -m http.server 8765 --directory dashboard/dist
```

Visit `http://localhost:8765`. The interface has seven sections: overview, data and exploration, adjusted associations, prediction, Ivy's validation, reproducibility, and exploratory model improvement. Filters change displayed results; the validation threshold slider does not change the frozen rules or test results.

To refresh the bundled results after rerunning the analyses:

```bash
.venv/bin/python dashboard/build_data.py
node dashboard/check.mjs
```

The builder reads the parent repository's `outputs/tables` directory. It also reads the Member 6 test log, when available, to display the recorded test count. `dist/assets/source-manifest.json` records source CSV fingerprints. No external JavaScript, fonts, analytics, or chart libraries are needed.

The display retains the study's limitations: accepted loans with resolved outcomes, unequal follow-up, non-causal associations, illustrative decision costs and dependence assumptions in bootstrap uncertainty. The contractual-maturity cutoff requires verification against the documented outcome-observation date.

The Site identity is in `.openai/hosting.json`. Publish a checked copy through Sites while preserving its private audience. Site source synchronization uses a separate temporary checkout; the dashboard remains an ordinary folder in the research repository.

Verification: source-level rendering and interaction checks plus HTTP response checks. Visual browser QA was unavailable because the browser access host could not be reached.

The improvement view preserves the original study alongside a separately labelled extension. Its aggregate tables come from `outputs/experiments/feature_extension`; rebuilding its curve samples also uses the local, ignored `data/cleaned/feature_extension_handoff/test_predictions.csv.gz` handoff produced by `analysis/11_validate_feature_extension.py`. No row-level handoff data is exported.

The improvement view also includes the optional XGBoost benchmark when its completed aggregate results exist. It reports all six development candidates and the frozen 2018 comparison, including uncertainty and probability calibration.
