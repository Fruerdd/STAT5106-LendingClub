# Generative AI use log

Record functional uses of generative AI, including brainstorming, code generation, debugging,
and documentation. Add entries when AI materially contributes to project work. Do not record a
human review or real-data analysis unless it actually occurred.

| Date | Function and contribution | Verification and limitations |
| --- | --- | --- |
| 2026-11-13 | Kimi Code CLI: project structure brainstorming; generated initial path handling, cleaning pipeline, analysis scaffolds, synthetic tests, CI workflow, and documentation. | `python3 -m pytest` run locally: 9 tests passed, including a synthetic CSV.GZ command-line test. The private real dataset was not available; no real-data cleaning, inference, model fitting, or validation results are claimed. Human code review has not yet been recorded. |
| 2026-11-18 | Cursor GPT-5.6 Sol: implemented prespecified feature preparation, Models A/B/C, maturity/unwinsorized/employment sensitivities, BH-FDR adjustment, term-level Wald tests, diagnostics, and an explicit chunked GLM HC3 score-sandwich covariance because statsmodels 0.14.x silently substitutes HC0 for GLM `cov_type="HC3"`; fitted the six real-data inference models and wrote auditable result tables. | The custom sandwich HC0 special case matched statsmodels HC0 with maximum relative covariance difference `5.74e-08`; 100,000-row smoke runs passed; all six full-data models converged in 5–6 iterations, with zero clipped leverage denominators and maximum VIF below 2.76. Outputs contain 265 coefficient rows and 119 term-test rows. A formal narrative report, forest plot, dedicated inference tests, predictive performance, and human review are still pending. |
| 2026-11-19 | Cursor GPT-5.6 Sol: added four dataset-independent inference tests, deterministic report generation, and a selected-association forest plot; generated `report/member4_inference_report.md` and the final figure from the audited tables. | All 13 cleaning/inference tests passed. A second complete six-model real-data run reproduced the coefficient CSV, global-test CSV, sample-summary CSV, preprocessing JSON, and PNG byte-for-byte; the Markdown report also matched after normalizing its intentionally relative figure path. Predictive metrics remain out of scope and human statistical review is still pending. |
| 2026-11-19 | Cursor GPT-5.6 Sol: performed a read-only pre-PR delivery review; added formula-metadata compatibility for statsmodels 0.14/0.15 and prevented partial scenario runs from overwriting the complete reviewed outputs; added regression tests for both safeguards. | All 15 tests passed with no linter or diff-format errors. A final complete six-model real-data run succeeded after the fixes. Human approval is not claimed; it must occur through the pull-request review requested from the downstream modelling owner and team. |

## Entry template

| Date | Function and contribution | Verification and limitations |
| --- | --- | --- |
| YYYY-MM-DD | Tool/model; brainstorm/code/debug/documentation task; files or decisions affected. | Commands/tests actually run and their results; reviewer and date if reviewed; remaining limitations. |
