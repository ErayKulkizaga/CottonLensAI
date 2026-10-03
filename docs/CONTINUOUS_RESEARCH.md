# CottonLens continuous research

## Run in Colab

Open `CottonLens_Continuous_Research_Colab_v2.ipynb` from the CottonLensAI Drive folder. Select a T4 GPU. Run the first five code cells: Drive, verified source extraction, locked environment, frozen research preparation, and the research round. The default is `ROUND = 'A'`, `EXTENSION = 0`.

The round performs `diagnose -> search -> compare`. This is a large nested search, not a one-minute smoke run: each candidate has three inner blocks across eight outer years, early stopping/refit, and shortlisted three-seed confirmation. Completed fits survive disconnection. Re-run the same cells with the same experiment/source to resume. Do not launch two writers. A stale `.writer-lock` may be removed only after verifying the previous process is stopped; preserve its contents for diagnosis.

Do not change source, data, dependencies, or protocol within a frozen experiment. Use a new `research-*` name when any identity changes. An interrupted fit is retried; a completed corrupt checkpoint raises an error. Reproduction uses a separate identity and fresh fits; cache hits are continuation, not new reproduction evidence.

## Implemented research rounds

- A: 128 CUDA XGBoost and 128 GPU CatBoost candidates per horizon, plus 24 Ridge and 24 ElasticNet references. Four target-scale/regularization diagnostics run first. Price MAE drives early stopping and inner selection.
- B: 64 candidates per horizon for XGBoost, CatBoost, and logistic classification. Down/flat/up probabilities and calibration remain separate from price-implied direction. Historical class-frequency/majority and median-return references use past labels only.
- C: 48 candidates per family/horizon for MLP, LSTM, TCN; windows 20/60/120 preserve source observation rows and explicit missingness.
- D: 32 candidates for the strongest tabular and sequence families on expanded causal OHLCV features. Run A and C first.
- E: histories 3/5/all years and refit cadence 5/21/63/126; locked tree counts within outer blocks. Up to three models form a nonnegative OOF-weighted ensemble.
- A/C extensions: `EXTENSION = 1`, then sequential numbers, 64 candidates for each of the top two families. Decisions use recorded inner scores only. No initial positive gain means diagnosis/new information; two consecutive gains below 0.5% stop widening that space.

Eight annual outer blocks end in 2016--2023, each with 126 eligible origins and three prior 63-origin inner blocks. Each scored inner block has a separate earlier early-stopping segment; its outcomes do not choose its own tree count. Labels must mature before refit. All results are **reused historical research evidence**, never an unseen holdout. Audit 2024+ is excluded from selection. New information with shorter coverage gets a separate cohort/experiment and cannot be ranked against the full-history cohort as if comparable.

Input snapshots preserve feature-missing dates. Train-fitted medians, standardization and missingness indicators are saved explicitly. The series remains a daily Cotton futures price proxy, not physical spot cotton. Decision-time convention is the next UTC day after the bar; exchange calendar and contract-roll cause are not independently verified.

## Outputs and controls

`experiments/research-v1/` contains frozen ready/history identities, diagnosis/quality, ledger/completed and checksummed model payloads, failed/interrupted attempts, per-fold decisions with **every trial**, outer predictions, comparison reports and a concise lessons packet. Each fit records data preparation, compute/checkpoint timing, GPU samples and peak memory, selected/run iterations, curves, prediction counts and train/test metrics. CPU reference estimators are explicitly labelled. GPU model fallback is an error. TensorFlow/CatBoost request an 80% memory ceiling; XGBoost records observed usage rather than claiming a hard allocator ceiling. Jobs are serial until measured throughput justifies concurrency.

Reports include paired MAE bootstrap with 20-observation blocks and 10/40 sensitivity, preserving fold boundaries. Gates: at least 5% MAE advantage, direction 53%/55%, and six of eight fold wins. Old four-fold gates remain unchanged. A failed price gate retains Naive.

## Lock, reproduce, export

After examining search results, run the final notebook cell with `STAGE = 'lock'`, then `'reproduce'`, then `'export'`. Lock closes that experiment to further search. Fresh historical reproduction must agree within 1e-6 log return. CatBoost GPU is nondeterministic; exceeding the tolerance blocks a verified release rather than relaxing the standard.

Schema 3 records feature order, training preprocessing, target inverse, member weights and model roles. Portable payloads use native XGBoost, ONNX or linear JSON, without TensorFlow/CatBoost in backend. Deployment refits and historical evaluation identities are separate. Export checks an independent CPU inference environment before writing its ZIP and checksum. The frozen source is used for the current refit; the output is not a claim to include bars newer than that snapshot. No automatic import into the old runtime/DB occurs.

The existing demo remains the prior validated release until a new Colab result passes export and application acceptance. Research direction models are reported in files; no classifier probability is silently substituted into the price API. Model Lab derives fold counts and wins from release metadata. Real GPU-release application acceptance remains required before demo promotion.

## New publication sources

CFTC, WASDE and contract-curve histories are **not enabled merely by this implementation**. `research/publications.py` accepts reviewed publication packages: actual per-report timestamp, as-published vintage, source hash and checksummed publication/vintage evidence are mandatory. General release calendars are insufficient. Use `--publication-package PATH` during prepare in a new research namespace; package coverage determines its separate comparison cohort. Paid data is not purchased. Unverified or revised historical releases are excluded.

## Prospective evidence

`python -m cottonlens_ml.research.prospective record --store NEW_STORE --input FEATURES.parquet --lock locked.json --artifact VERIFIED_ARTIFACT --repo REPO` records a locked release's forecast during the next-day decision window. Its input snapshot and prediction are immutable; backdating and mixed locks/releases fail. Supply current causal features with the release's full feature order. No scheduling or automatic fresh-data retrieval is implied.

`... prospective score --store NEW_STORE --input COTTON_OUTCOMES.parquet` with date/close columns returns pending until exactly the first 126 recorded origins mature through T+5. Missed source origins are explicitly reported. No partial performance is exposed for candidate tuning. These commands require actual subsequent observations; synthetic tests are not prospective evidence.

## Local verification and preserved state

Only unit, contract and synthetic no-fit tests run locally. Backend conftest creates an isolated temporary database before importing app configuration. Original `backend/cottonlens.db`, preservation backup, R2 and sprint ZIP hashes are checked; original DB recovery is still unverified and no recovery attempt is made.

GPU capability, actual market performance, fresh GPU reproducibility and converted GPU-model CPU parity can only be confirmed by Colab execution. Passing local tests does not assert these outcomes.

## TF device placement bugfix

### V2 notebook supervision and interrupted runtimes

`ml/notebooks/cottonlens_research_colab_recovery.ipynb` resumes the frozen V2 ZIP and `research-v2-tf-placement` identity. Its separately checksummed `ml/colab_progress.py` helper changes process supervision only. Logs go to local disk and a daemon mirrors unique logs to Drive; notebook heartbeat reports process CPU, wait channel, last output and last successful log copy even when the engine is silent. It does not eliminate blocking Drive operations inside the frozen engine. A saved-fold count is not a time estimate; candidate enumeration is followed by seed validation and outer fitting.

Delete the previous Colab runtime before recovery. Legacy `.writer-lock` has no owner or lease: the notebook requires explicit confirmation that no other runtime is running this experiment, checks local engine processes, and removes only an empty lock. Never delete checkpoint folders to resume. Completed records are checksum-validated by the frozen engine; interrupted fits may repeat. Notebook stop terminates its subprocess. A hard runtime kill can still leave a legacy lock. Supervision is verified locally with print/sleep subprocesses, not model training.

Research v1 stopped before market search: Keras built the CPU-only RangeDataset inside the forced GPU device scope. V2 keeps model construction on GPU, leaves Keras fit/data-adapter/save orchestration outside that scope, and checks trainable-variable and output devices while keeping strict placement. Source `cottonlens-research-source-v2.zip` uses experiment `research-v2-tf-placement`; four completed diagnostic controls in research-v1 remain immutable and rerun only as bugfix verification. Colab GPU execution must confirm the correction; local tests do not train models.
