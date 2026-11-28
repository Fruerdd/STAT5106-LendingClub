# Reducing missed defaults

A missed default is a loan predicted fully paid that actually defaults. The comparison uses the existing ensemble scores for the same 269,620 frozen test loans, including 58,758 defaults.

The active rule is the maximum-F1 cutoff from experiment 18. Its test default F1 is **0.45219**. The 99.5% validation recall cutoff reached **99.45% test recall** with **323 missed defaults** during the high-recall experiment.

| Rule | Score cutoff | Missed defaults | Defaults caught | Fully paid loans flagged | Default recall | Share of loans flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Maximum validation F1, active | 0.309770 | 19,910 | 38,848 | 74,214 | 66.12% | 41.93% |
| Previous precision-focused rule | 0.336095 | 23,467 | 35,291 | 63,280 | 60.06% | 36.56% |
| 95% validation recall | 0.139383 | 2,877 | 55,881 | 162,634 | 95.10% | 81.05% |
| 99% validation recall | 0.076393 | 606 | 58,152 | 192,159 | 98.97% | 92.84% |
| 99.5% validation recall | 0.060793 | 323 | 58,435 | 198,401 | 99.45% | 95.26% |
| 100% validation recall | 0.016428 | 3 | 58,755 | 210,817 | 99.995% | 99.98% |
| Flag every loan | 0 | 0 | 58,758 | 210,862 | 100% | 100% |

The 99% validation target reduced missed test defaults by 97.4% and flagged 128,879 additional fully paid loans. The 100% validation target spared just 45 fully paid test loans from a flag and still missed three defaults. Flagging every loan gives zero missed defaults by construction and classifies every fully paid loan incorrectly.

For each target, the script maximizes precision among cutoffs that meet the specified recall on validation. It writes the rules before opening the test scores. The same fitted ensemble is used for all cutoffs, so the comparison measures the trade-off between missed defaults and false alarms at different operating points. Earlier experiments examined these test loans.

Run `.venv/bin/python analysis/20_default_recall.py` after experiments 18 and 19. The exact cutoffs are in the [frozen rules](../outputs/experiments/default_recall_602020/frozen_rules.json). The full confusion matrices and metrics are in the [test table](../outputs/experiments/default_recall_602020/test.csv).
