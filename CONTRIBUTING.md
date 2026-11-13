# Contributing

## Branch and review workflow

1. Update your local `main`, then create one branch per member/task, for example
   `git switch -c member3-eda` or `git switch -c pavel-logistic-model`.
2. Make small, focused commits with clear messages. Keep unrelated formatting and generated
   artefacts out of the change.
3. Run `python -m pytest` from the repository root before requesting review. Run any affected
   analysis against the frozen real dataset and distinguish those results from synthetic tests.
4. Open a pull request and obtain team review before merging into `main`. Review data leakage,
   temporal-split behavior, train-only preprocessing, and reproducibility, not just style.
5. Document dataset, cohort, feature, split, or model changes in the pull request and relevant
   report files. Notify downstream owners when an input contract changes.

## Data and artefacts

- Never commit or push raw data, cleaned data, compressed datasets, or large model artefacts.
- Every member must use the same frozen
  `data/raw/accepted_2007_to_2018Q4.csv.gz` version. Do not overwrite it with a different extract.
- Keep `data/raw` and `data/cleaned` local. A private GitHub repository is not an exception to
  this rule. Check `git status --short --ignored` before committing when in doubt.
- Small, non-sensitive result CSVs in `outputs/tables` may be reviewed and committed. Do not add
  a global `*.csv` ignore rule.
- Do not commit access tokens, credentials, personal absolute paths, `.venv`, or `.idea`.

## Analysis standards

Use `src/paths.py` rather than machine-specific paths. Preserve the resolved-outcome definition
and leakage allowlist unless a reviewed project decision changes them. Fit imputers, scalers,
encoders, feature selection, and threshold choices on training data only. Keep test data for
out-of-time evaluation. Never generate synthetic findings or figures for the real project.

Record functional generative-AI use in `report/ai_use_log.md`, including brainstorming, code
generation, debugging, and documentation. Record the verification actually performed and leave
human review status honest.
