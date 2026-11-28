# Exploratory predictive feature extension

This extension investigates whether additional information improves precision and recall after the original nine-candidate follow-up produced no meaningful 2018 ranking gain. It applies only to a separate predictive experiment. The original inference specifications, primary predictive results and Ivy's validation remain unchanged.

The outcome and resolved-loan cohort remain fixed. Tune on 2016 loans using earlier training loans; choose candidates by 2016 average precision. Refit the frozen choice on loans before 2017. Select operating thresholds on 2017 only, then evaluate on the already examined 2018 cohort. All results are exploratory because development follows inspection of the original validation/test results.

## Additional information

Only fields already retained by the cleaning allowlist and defined in the supplied `Data_Dictionary.xlsx` are used:

| Source | Predictor | Rationale |
| --- | --- | --- |
| loan_amnt | log1p loan amount | Preserve the amount independently of the existing income ratio |
| installment, annual_inc | Monthly payment / monthly income | Origination payment burden; nonpositive income gives missing |
| revol_bal | log1p revolving balance | Credit balance amount complements utilization |
| total_acc | Total credit accounts | Credit-file breadth |
| open_acc, total_acc | Open / total accounts | Credit-file composition; nonpositive total gives missing |
| mths_since_last_delinq | Months since last delinquency | Preserve history and structural missingness through native missing handling |
| mths_since_last_record | Months since last public record | Preserve history and structural missingness through native missing handling |
| addr_state | State category | Borrower-provided application geography; descriptive subgroup limitations remain |

Three separate lender-proxy specifications use interest rate, grade or subgrade. They never combine these three fields. The supplied dictionary describes subgrade as the LC-assigned loan subgrade; LendingClub's 2011 SEC prospectus also documents A1–G5 grades assigned to loan requests: https://www.sec.gov/Archives/edgar/data/1409970/000095012311058394/c18687e424b3.htm.

Original engineered numerical features retain training-fitted imputation/caps. Additional numerical features use the boosting model's native missing handling; no outcome-based imputation or global caps are added. Category codes are learned from training only and unknown levels map to missing. The original credit-event-date exclusion was a primary inference/prediction contract choice; this explicitly labelled extension revisits it for prediction without changing inferential claims.

The registered candidates are expanded interest-rate models with 15 or 31 leaves, an expanded grade model with 31 leaves, and an expanded subgrade model with 31 leaves. All use learning rate 0.05, at most 800 iterations, minimum leaf size 100, seed 5106 and training-only early stopping. L2 penalty is 10 for the 15-leaf model and 5 for the others.

No repayment, recovery, settlement, hardship, updated FICO or last-payment fields may enter the predictor matrix. Other raw-dataset fields are not added because their timing has not been established for this experiment. Geographic associations do not establish causality or fairness, and this experiment is not a lending deployment.
