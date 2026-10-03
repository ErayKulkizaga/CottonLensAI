# Research status — 3 October 2026

This repository contains an interview/demo application and a reproducible research system. Historical experiments have not established the fixed price-performance gates. Naive remains the primary price reference. These results do not demonstrate superiority over Cotcast; its target, forecasts and evaluation protocol are unavailable for a common comparison.

## Completed evidence

The full-year protocol evaluates 2016–2023, eight annual outside blocks and 2006 common origins per horizon. Research history has already been examined; it is reused historical evidence. The 2024+ seen audit never selects candidates.

| Selected strategy | T+1 Naive MAE gain | T+5 Naive MAE gain | Winning years T+1 / T+5 |
|---|---:|---:|---:|
| Full-history, six small recipes with past-only shrinkage | -0.262% | -0.996% | 1/8 / 2/8 |
| Same recipes, fixed three-calendar-year history | -0.198% | -1.477% | 2/8 / 1/8 |

Nine source-channel pilots covered 63 feature groups and 434 saved group/year results. AMS and WASDE used separate shorter cohorts and cannot be ranked against the eight-year results. The tested USDA/CFTC/FX/crop/weather/On-Call recipes did not meet their contribution conditions. Most historical publication/vintage evidence remains Tier A: exploratory, excluded from release claims.

The normalized interval pilot also failed its research-priority conditions. HAR/GARCH mean QLIKE differences were positive but uncertain; they are not verified volatility superiority.

## What the latest diagnosis supports

Reconstructing 48 selected inner blocks from 162 existing fit records showed that winning all three inner blocks was insufficient for transport. The 2017 T+1 selection gained 1.472% inside but lost 1.518% in the next year. The 2018 T+5 selection gained 4.379% inside but only 0.235% outside. A retrospectively stricter guard alone yielded only 0.0249% T+5 gain; it is a diagnostic, not a new validated candidate.

Recency raw forecasts also lost MAE. Removing shrinkage or expanding the same history-length search is unsupported. The findings do not establish that all model families or all Cotton information are useless.

## Completed bounded comparison

`return-path-pilot-v1` compares a fixed Ridge(alpha=10) with 24 existing features against the same model with 59 preceding Cotton returns added. The current return is already in the base, so the challenger represents a full 60-observation path using 83 columns. There is no alpha/window/seed search.

Both arms preserve the snapshot, exact origins, full training history, train-only preprocessing/target scaling, five-observation purge, strict label maturity, annual past-only weight selection and 21-observation refit. Maximum 676 fits and 32 annual outputs, one CPU process/two threads, at most 60 minutes per session. Results determine whether this representation deserves more work. The design ID is `8c55cebb8867e1fd269b51e66b7e75f897d04dbe17dc4f8add11f29416932a94`.

All 676 fits and 32 outputs completed on the approved local CPU path. Each horizon has the same 2006 origins.

| Fixed Ridge arm | T+1 Naive MAE gain | T+5 Naive MAE gain | Winning years T+1 / T+5 |
|---|---:|---:|---:|
| 24-feature matched control | -0.323% | -1.087% | 0/8 / 0/8 |
| 83-feature return-path challenger | -0.043% | -0.952% | 0/8 / 0/8 |

The challenger lost inside against its matched control: -0.163% / -0.067%, with 0/8 / 2/8 inner year wins. Both research-priority decisions and price gates are false. The slight outside improvement over the control does not establish advantage over Naive. The two paired hypotheses have BH-adjusted p-values 0.0766 and 0.6922; these are reused-history evidence, not independent tests.

Annual past-only shrinkage selected zero weight for many years. The challenger has 1755 / 1001 flat forecasts; all-origin direction accuracy is 6.48% / 23.98%, compared with historical majority-direction references 48.75% / 52.74%. Flat means predicting no price change; these low all-origin accuracies are not a coin-toss experiment. They still fail the unchanged direction gate.

All 2028 fit payload hashes were verified. Cached replay added zero fits. Seventy-seven small checksum-verified packages were copied to the Drive directory; a clean metadata-only restore used 34 packages and reproduced the corrected report exactly without model downloads. This is replay/report verification, not fresh model reproduction or confirmation of cloud synchronization. A reporting-only correction changed `majority_direction_pct` from fraction to percentage units; the original report, fit source bundle and completed fits remain preserved. Current read-only workflows use `source-return-path-report-v2-20261003.zip`; prepare/pilot retain the original frozen fit bundle.

Decision: do not expand the same linear lag-window search or release the challenger. Naive stays primary. The next bounded task is diagnosis of the saved unshrunk challenger predictions against the matched control, separating transport failure from a representation that contains no usable linear signal. This negative comparison does not justify a broader nonlinear/GPU search.

### Raw-forecast diagnosis and next feasibility check

That diagnosis is complete, with zero fits. Reconstructing 288 signed inner fit receipts verified all selected yearly inner scores/weights. Raw path forecasts lost MAE both inside (-5.026% / -7.167% versus Naive) and outside (-4.424% / -5.971%). Outside direction accuracy was 49.45% / 50.05%; Spearman IC -0.02695 / -0.01303. Actual OOS log-return R² versus zero return was negative (-0.0525 / -0.0975). Removing shrinkage is not a remedy. These statistics describe this fixed linear model, not every model family.

Raw path losses concentrate in small realized movements (-23.01% / -29.70% MAE gain within that diagnostic bin). Bins use past-training quantile boundaries, but membership depends on future realized movement: this is explanatory evidence, never a rule for removing origins or predicting an investable regime. All 2006 origins remain in the score. Repeated inner dates are not independent observations.

The next single preregistered hypothesis is `agri-transfer-pilot-v1`: a shared Cotton/corn/soybean model against a matched Cotton-only model. The earlier crop ablation added crop feature columns; this comparison adds supervised auxiliary examples under shared coefficients. It does not assume that correlated markets provide three times the independent information or that shared training will improve forecasts.

Feasibility inputs contain 10568 recorded rows (3520 Cotton, 3524 corn, 3524 soybean) and six preceding-observation return/volatility/SMA features. Missing rows stay present; missing labels are never filled. The same 2006 Cotton origins and targets were verified. Historical crop publication/roll/vintage metadata remains unverified, so this is Tier-A exploration and cannot support release even if its exploratory score is positive.

Fixed Ridge(alpha=10), seed42, full history, annual past-only shrinkage, five-Cotton-observation purge and 21-observation refit; no basket/window/alpha/mixture/cadence search. Both arms use preprocessing/target transforms learned on mature Cotton training only. The pooled objective has fixed 50% Cotton / 25% corn / 25% soybean total weights, normalized to the control's total weight. Maximum 676 fits/32 outputs. Design ID `752a333dfdcbfd34f28467128c48940506b96914cd4a88b0e1c1d7291607b7c2`.

The same-ledger adapter and weighted-fit, temporal and resume contracts now pass. All 676 small CPU fits and 32 annual outputs completed, with the same 2006 origins per horizon. The preserved reference's terminal labels reaching 2024 are masked in this new preparation; the old snapshot is unchanged and those audit outcomes never enter this fit.

| Six-feature matched Ridge arm | T+1 Naive MAE gain | T+5 Naive MAE gain | Winning years T+1 / T+5 |
|---|---:|---:|---:|
| Cotton-only control | -0.025% | -0.092% | 3/8 / 2/8 |
| Shared Cotton/corn/soybean | +0.062% | +0.176% | 5/8 / 4/8 |

Shared training lost inside against its matched control: -0.0206% / -0.1124%, with 4/8 inner wins each. Neither research-priority signal nor price gate passed. Paired matched-control confidence intervals include zero; BH-adjusted p-values are 0.2216 / 0.2050. These small historical outside gains do not justify enlarging the same panel/mixture search.

All-origin direction is 44.37% / 39.23%, with 252 / 497 flat forecasts; majority-direction references are 48.75% / 52.74%. This is price-implied direction with flats, not a binary classifier. Naive stays primary and Tier-A data blocks release.

All 2028 payload hashes, saved transform matching, fixed asset-weight totals and refit label/purge boundaries were verified. Cache replay added zero fits. Seventy-seven checksum batches were copied to the Drive directory; a clean 34-package metadata-only restore reproduced the exact report. This verifies replay, not fresh reproduction, GPU training, deployment parity or cloud synchronization. Compute receipts total 20.017 seconds, excluding runner/report/transfer wall time.

The existing Research Workbench and frozen `source-agri-transfer-pilot-v1-20261003.zip` support status/prepare/pilot/compare; Run All remains read-only. No new notebook was created. Local evidence: `output/reports/agri-transfer-implementation-20261003/completion.md`; report `transfer-d0a1908023824705.json`. Next: examine saved raw shared-model forecasts without new fits before selecting a distinct hypothesis. Do not retune asset weights, alpha or cadence on outside results. The completed return-path source packages and experiment are unchanged.

The existing Experiment/Ledger and two workbenches are used. New metadata catalogues allow status/compare to restore preparation and result batches without scanning model packages. Resume verifies completed work; incomplete or corrupt checkpoints cannot silently become completed experiments. Run All defaults to read-only status. GPU/sequence training stays in Colab. The isolated CPU group does not install TensorFlow/CUDA into the backend.

## Evidence and operating limits

- Price gate: at least 5% MAE improvement, T+1 53% / T+5 55% all-origin direction accuracy, at least 6/8 yearly wins. Flat forecasts are reported; conditional nonflat accuracy never replaces this gate.
- Paired bootstrap: 10000 replicates, fold boundaries preserved, main block length 20 and sensitivity lengths 10/40. Post-hoc power assumptions and reused-history selection limits remain explicit.
- The Windows collector records prospective Naive/EWMA forecasts and source vintages with cutoff/missing/no-backfill rules. Computer uptime and local Drive copy verification do not establish continuous service or cloud synchronization.
- New sources require specific availability/vintage evidence for Tier B. Public access alone does not establish redistribution rights. Data budget is zero; paid resources need separate approval.
- Old databases, preservation backups, snapshot/release identities and the 24-column production artifact remain protected. The 83-column challenger cannot enter runtime without exporter/importer/runtime/parity contracts being upgraded together.
- All ML tests passed locally in the isolated CPU environment: 472 passed, one skipped. Changed Python files pass Ruff. Prior GitHub backend, frontend, ML data contracts and Compose checks passed; the latest commit's CI status must be checked separately. This does not verify GPU training or Linux Colab execution. A prior broad Ruff scan reported six import-order findings in unchanged test files.

Canonical decisions: [MASTER_PLAN_20261001.md](MASTER_PLAN_20261001.md). D21 remains 22 October 2026 at 20:06:59 UTC; adding a new experiment does not restart that date. Local data, model files, credentials and private handoff material are excluded from Git.
