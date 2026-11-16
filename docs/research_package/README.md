# CottonLensAI research package

This package is the complete, self-contained record of the CottonLensAI research project, written so that any question from an ML engineer, quant or data scientist can be answered from it. It covers both phases:
- **Phase 1** (21 Sep – 7 Nov 2026, built with Codex): cotton futures price and risk forecasting.
- **Phase 2** (13 – 16 Nov 2026, built with Claude): US upland cotton production forecasting against USDA.

Every number here comes from the project's own ledgers, evidence files and documents. Where a number is an estimate, it says so. Nothing in the package changes the original evidence.

## How to read it

| File | What it answers |
|---|---|
| [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) | What the project is, the timeline, the headline results in one page |
| [02_NUMBERS_CHEATSHEET.md](02_NUMBERS_CHEATSHEET.md) | Every countable number (fits, trials, scenes, pixels, reports) with its source |
| [03_PHASE1_PRICE_AND_RISK.md](03_PHASE1_PRICE_AND_RISK.md) | Phase 1 study by study: what was tried, how, with which settings, and what came out |
| [04_MODEL_FAMILIES_AND_HYPERPARAMETERS.md](04_MODEL_FAMILIES_AND_HYPERPARAMETERS.md) | Every model family: how many fits, where it was used, exact hyperparameters, results (for example "what did the 14,723 XGBoost fits try?" or "what was the LSTM?") |
| [05_PHASE2_PRODUCTION_VS_USDA.md](05_PHASE2_PRODUCTION_VS_USDA.md) | Phase 2: all 45 trials, satellite pipelines, the GPU CNN, sealed tests, the momentum rule, the 2026 forward lock |
| [06_DATA_SOURCES.md](06_DATA_SOURCES.md) | Every data source, its volume, and the rule for when it may be used (leakage control) |
| [07_EVALUATION_PROTOCOL_AND_METRICS.md](07_EVALUATION_PROTOCOL_AND_METRICS.md) | Metrics (MAE gain, MAPE, QLIKE, direction), gates, rolling origin, purges, shrinkage selection, bootstraps, preregistration, sealed holdouts |
| [08_FAILURES_AND_LESSONS.md](08_FAILURES_AND_LESSONS.md) | What failed, why, and what it taught; bugs and leaks found and fixed |
| [09_INTERVIEW_QA_TR.md](09_INTERVIEW_QA_TR.md) | Interview questions and answers in Turkish |
| [10_GLOSSARY.md](10_GLOSSARY.md) | Terms in English and Turkish |
| [11_QUESTIONS_FOR_CODEX.md](11_QUESTIONS_FOR_CODEX.md) | Open gaps in the Phase 1 record, as questions for Codex (Turkish) |
| [13_LITERATURE_AND_CONTRIBUTION.md](13_LITERATURE_AND_CONTRIBUTION.md) | Prior work (smoothing in USDA forecasts, 1987–2023) and what this project adds; robustness and comparison with a published-style adjustment |
| [15_USDA_ERROR_MECHANISM.md](15_USDA_ERROR_MECHANISM.md) | Where USDA's cotton error comes from: harvested area vs yield, and the 2012 shift in Texas (T48) |
| [14_DECISIONS_AND_COMPUTE.md](14_DECISIONS_AND_COMPUTE.md) | Who decided what (owner vs AI tools), and recorded compute and cost |
| [12_CODEX_ANSWERS.md](12_CODEX_ANSWERS.md) | Codex's sourced answers (16 Nov 2026): LSTM architecture, pre-ledger fit counts, forward-test status, line-by-line corrections. The corrections are applied in 01–04, 08 and 09 |

## Machine-readable data (`data/`)

| File | Rows | Content |
|---|---|---|
| `codex_fit_recipes.csv` | 569 | Every ledger record of Phase 1: profile, kind (market / synthetic / audit), completed fits, horizon, model family, task, target, loss, window, cadence, seed, params JSON, roles, full feature list |
| `codex_fits_pivot.csv` | 73 | Completed fits by profile × family × kind |
| `codex_profiles.csv` | 41 | One row per research profile: fits, families, horizons, losses, targets, feature counts, distinct parameter sets, scored results, documents |
| `codex_experiments.csv` | 239 | Registry experiments with selected and raw MAE, gain vs naive, direction, active rate, years |
| `codex_market_v1_ablation.csv` | 16 | `research-market-v1` ablation results (20,360 sub-fits; not in the fit ledger) |
| `feature_catalog.csv` | 487 | Every feature name used in Phase 1, how many recipes used it, and in which profiles |
| `hyperparameter_ranges.csv` | 27 | Per family and parameter: count, min, median, max, distinct values |
| `docs_catalog.csv` | 127 | Every project document with its title, date and result/decision lines |
| `evidence_index.csv` | 297 | Every evidence file with size and SHA-256 |
| `claude_phase_trials.csv` | 36 | Phase 2 trials T01–T46 with scores and verdicts |
| `momentum_by_year_2002_2025.csv` | 186 | The momentum rule vs USDA, per report month, region and year |
| `bias_correction_t25_by_year.csv` | 48 | The pooled bias correction vs USDA per year (2013–2025) |
| `data_versions.csv` | 135 | Every cached input file behind the Phase 2 results: SHA-256, size, retrieval time, timing rule |

`images/` holds real Landsat NDVI renders of Lubbock County, Texas (2021, 2022, 2023) and the 2021 Cropland Data Layer for the same window. The scripts that produced them are in `research/package/`.

## Reproducing the results
```
python research/reproduce.py
```
This re-runs the seven Phase 2 scorers (momentum record, 2026 lock, robustness, T46, T47 stage A, both error-mechanism studies) on the cached public data. It compares every number with the committed evidence JSONs, and exits non-zero on any difference. On 16 Nov 2026 all seven matched under the pinned environment (`research/requirements-phase2.txt`: numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1) and under numpy 2.3.2 / pandas 3.0.1. The county satellite models (T01–T43) and the CNN need the large satellite feature caches and Kaggle; their outputs are kept as evidence files but are not re-run by this script.

## Regenerating the data

```
PYTHONPATH=<pinned env> python research/package/build_catalogs.py
```
The catalogs are rebuilt from `research/trials.json`, `research/registry.json`, `docs/` and `research/evidence/`. The Phase 2 CSVs come from the evidence JSONs named in each section.

## Related public material
- Interactive research page (English): `docs/portfolio/cotton-usda-research.html`
- Trial ledger (Phase 2): `docs/CROP_NOWCAST_TRIALS.md`
- Sealed test record: `docs/SEALED_TEST_T39_PREREG_20261115.md`
- Forward lock: `docs/FORWARD_LOCK_2026.md`
- What was tried: `docs/TRIED_AND_LEARNED_20261116.md`
