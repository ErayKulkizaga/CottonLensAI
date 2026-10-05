# CottonLens model experiment log

## v1 — completed, gates not met

Evidence: `cottonlens-results-v20260923-0652.txt`, produced in Colab from commit `c33d6ae`.
Dataset: 2010-04-01 to 2026-09-22, 3,905 labeled origins, four 126-observation evaluation blocks.

| Candidate | T+1 MAE gain vs Naive | T+1 direction | T+5 MAE gain vs Naive | T+5 direction |
|---|---:|---:|---:|---:|
| XGBoost | -0.376% | 52.78% | +0.725% | 54.56% |
| LSTM | -1.094% | 51.98% | -0.011% | 52.78% |

Both horizons selected Naive. The previously viewed 2024+ audit also favored Naive.
The best LSTM tuning checkpoint was epoch 1: train/validation loss moved from
0.73125/0.92163 to 0.63588/0.97850 by epoch 11. More epochs alone are not supported by this evidence.
Macro/regime additions did not consistently reduce the fixed ablation model's MAE.

Confirmed implementation findings:

- A percentage-based inner validation period withheld about two recent years, with no subsequent refit. Fold 1 predicted April 2022 onward using weights fitted only through April 2020.
- XGBoost default early stopping optimized log-return RMSE, while configuration selection optimized price MAE.
- A saved best-so-far LSTM checkpoint was treated as a completed experiment even if interrupted before training history was written.
- A missing target origin could compress LSTM feature context, even though its past features were available.
- The report omitted fold/audit sample counts and labeled zero-based best iteration as a tree count. Balanced accuracy silently treated a zero forecast as 'down' while direction accuracy treated it as flat.

These findings identify correctable limitations. They do not prove that the available features contain enough predictive signal to meet the gates.

## corrected V2 — Day 1 protocol, new Colab results pending

- Fixed 126-observation inner validation; keep T+5 purges.
- Lock settings/tree count/epoch count on that validation, then refit using available train + validation labels strictly before the evaluation period.
- Fit scalers separately for tuning and refit. Preserve 60-step, two-output LSTM architecture and the eight/four configuration budgets.
- Use price MAE for XGBoost early stopping. Refit Ridge and ablation models consistently.
- Reset seed 42 before each LSTM initialization; fingerprint recipe/source/data, and reuse only completed checkpoints. Interrupted LSTM trials restart safely; completed trials remain cached.
- Use actual historical feature windows when a target is absent. Report remaining missing-feature calendar gaps explicitly.
- Report prediction direction/spread, train/validation drift, sample counts, paired MAE gain intervals and actual refit dates. Retain both v1 and v2 evidence.

Hypothesis: recent refitting and metric alignment improve generalization. Outcome remains unmeasured until Colab runs v2.
No changes to acceptance thresholds. No local or CI model training.

## Fixed acceptance criteria

- Price MAE gain = `100 × (Naive MAE − model MAE) / Naive MAE`; minimum 5%.
- Direction accuracy: correct sign of T+1/T+5 log return; minimum 53%/55%.
- Lower MAE than Naive in at least 3 of 4 periods.
- LSTM replaces an eligible XGBoost only with at least 5% further MAE improvement and no worse direction accuracy.
- The retrospective majority-direction reference is not a deployable trained baseline. Naive abstains on direction by predicting zero return.

The evaluated folds have now informed development. The 2024+ audit was already observed and is descriptive only: it cannot select, veto or tune a candidate. Neither is an independent future test; forward-recorded predictions must establish future generalization. A gate pass alone is not proof of profitability or statistical significance. The earlier V1 text report is available, but its raw/processed snapshots, exact 504 origins, tracking/checkpoints and full release ZIP are not locally verified. Corrected V2 freezes a separate data identity and 504-origin cohort; its metrics cannot be described as exact V1 replication. The 126-observation refit cadence is the locked evaluation policy; 21 observations remains an unvalidated development-period hypothesis and is not a Day 1 tuning axis.
