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

The existing Experiment/Ledger and two workbenches are used. New metadata catalogues allow status/compare to restore preparation and result batches without scanning model packages. Resume verifies completed work; incomplete or corrupt checkpoints cannot silently become completed experiments. Run All defaults to read-only status. GPU/sequence training stays in Colab. The isolated CPU group does not install TensorFlow/CUDA into the backend.

## Evidence and operating limits

- Price gate: at least 5% MAE improvement, T+1 53% / T+5 55% all-origin direction accuracy, at least 6/8 yearly wins. Flat forecasts are reported; conditional nonflat accuracy never replaces this gate.
- Paired bootstrap: 10000 replicates, fold boundaries preserved, main block length 20 and sensitivity lengths 10/40. Post-hoc power assumptions and reused-history selection limits remain explicit.
- The Windows collector records prospective Naive/EWMA forecasts and source vintages with cutoff/missing/no-backfill rules. Computer uptime and local Drive copy verification do not establish continuous service or cloud synchronization.
- New sources require specific availability/vintage evidence for Tier B. Public access alone does not establish redistribution rights. Data budget is zero; paid resources need separate approval.
- Old databases, preservation backups, snapshot/release identities and the 24-column production artifact remain protected. The 83-column challenger cannot enter runtime without exporter/importer/runtime/parity contracts being upgraded together.
- All ML tests passed locally in the isolated CPU environment: 458 passed, one skipped; after the reporting correction, 14 focused runner/notebook regression checks also passed. GitHub backend, frontend, ML data contracts and Compose checks passed. This does not verify GPU training or Linux Colab execution. Changed Python files pass Ruff; a broad scan also reported six pre-existing import-order findings in unchanged test files.

Canonical decisions: [MASTER_PLAN_20261001.md](MASTER_PLAN_20261001.md). D21 remains 22 October 2026 at 20:06:59 UTC; adding a new experiment does not restart that date. Local data, model files, credentials and private handoff material are excluded from Git.
