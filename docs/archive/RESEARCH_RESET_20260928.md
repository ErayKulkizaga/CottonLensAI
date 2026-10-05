# Research reset — 2026-09-28

## Verified position

Source: `output/reports/research-v2-A-saved-results.json`, 64 decisions and 64 outer predictions, 1,008 origins per horizon across eight reused historical folds. Negative improvement means worse than Naive. All price-model gates failed.

| Model | Horizon | MAE improvement vs Naive | Direction | Fold wins |
|---|---:|---:|---:|---:|
| xgboost | 1 | -0.611% | 46.92% | 1/8 |
| catboost | 1 | -0.463% | 48.31% | 0/8 |
| ridge | 1 | -2.135% | 47.42% | 2/8 |
| elasticnet | 1 | -0.819% | 47.12% | 1/8 |
| xgboost | 5 | -0.528% | 55.26% | 2/8 |
| catboost | 5 | -1.166% | 52.58% | 3/8 |
| ridge | 5 | -6.340% | 50.00% | 3/8 |
| elasticnet | 5 | -3.575% | 50.79% | 4/8 |

The new `research-free-data-v1` has zero admitted publication packages. Eight synthetic controls passed on CUDA; six real-data pilot fits and all 18 payload files were verified independently on Drive. Pilot relative inner MAE is 1.0000112317: essentially Naive, on one recipe/fold only. It cannot select a model or justify abandoning an entire family.

Compact ledger accounting for these 14 fits: compute_seconds=24.1451, checkpoint_copy_seconds=1.1743, data_seconds=0.2734. Compute includes fit overhead, not measured GPU-active or billed hours. Older research-v2 has no compact timing index; total historical GPU hours remain unknown. Do not equate notebook elapsed/wait time with GPU work.

## Diagnosis and changed priorities

1. Engineering friction was real: fragmented notebooks, misleading granularity, CPU-only sprint, and the pilot verifier using experiment root instead of ledger root. The latter was a verifier bug; six model payloads were not lost.
2. Synthetic learning establishes an executable learner, not that current Cotton information predicts future price changes. Short-horizon Naive is a serious persistence benchmark; MAE and directional classification are different tasks.
3. Earlier small inner gains did not transfer to outer periods. Candidate-selection noise, regime shifts and limited predictive information are hypotheses, not established single causes. A senior review examines per-year residuals, prediction spread, train-validation gap and actual feature availability before enlarging search.
4. Added raw USDA/ECB files have not added model information yet. Historical release/vintage provenance is the binding data dependency; record explicit source-ready criteria instead of collecting indefinitely.

## Two persistent workbenches

- `ml/notebooks/cottonlens_data_workbench.ipynb`: raw acquisition/inventory and source status. It does not certify historical release clocks. Complete AMS/NASS/FAS provenance in existing source modules before package admission; never silently fill a guessed publication timestamp.
- `ml/notebooks/cottonlens_research_workbench.ipynb`: status, diagnose, full feature ablation, gated tabular search, direction/sequence/refinement, reports, explicit lock/reproduction/export. Default run-all now selects status; this research phase blocks training without an admitted publication package. Existing 14 fits are reused by identity. No more per-candidate notebook deliveries.

The current bundle stays c669ca9a... to preserve existing experiment identity. Newly verified data or training-code changes require a new immutable bundle/experiment, configured in the same workbench. TFT, full climate/text acquisition and production schema-v4 are not complete; do not present stage selectors as proof that the entire long-term roadmap is delivered.

## Execution sequence and decision gates

1. First finish source publication/vintage review for at least one admissible package and freeze its new experiment identity. Then complete the four-group, two-family, two-horizon, eight-fold comparison as one resumable workload. Base, no-volume Cotton, expanded features and expanded availability share origins and an equal 16-candidate budget. About 20,352 underlying fits are possible including confirmations/refits; cached equivalents reduce new work. Report counters as work units, never elapsed percentage. Do not extrapolate 13.6 seconds from one early-stopped recipe to the whole run.
2. Review matched inner contributions (>=0.5% mean improvement and >=5/8 positive inner blocks) and full outer error/coverage patterns. Maintain per-fold past-only selection. Results remain reused research evidence; no tuning on 2024+ audit.
3. In parallel engineering work, finish at least one verified USDA information package. If provenance cannot be established, mark it blocked and move to a verifiable source; no broad search on unidentified revision history. Keep raw files private when redistribution terms are unresolved.
4. Broader Optuna tabular search follows admitted new information and complete ablation. Preserve 128-trial budgets, three seeds and complete inner validation. Use L4 initially as an operational choice, not a proven cost optimum. Compare measured throughput/compute/copy time before paying for A100; sequence work can justify its memory. GPU availability does not create additional independent historical samples.
5. Run sequence, separate direction, history/refit and OOF ensemble only as recorded hypotheses. Two consecutive rounds below 0.5% inner improvement redirect work rather than repeating the same space.
6. Freeze candidates, retain 5% MAE / 53%-55% direction / 6-of-8 gates, 10,000 paired bootstrap replicates, fresh reproduction and CPU parity. Keep immutable prospective predictions for 126 matured origins. No success guarantee or Cotcast superiority claim without common evaluation.

## Presentation and archive policy

Present one source registry, an experiment table including failures, matched ablations, uncertainty, reproducible release and application. Notebook count is not evidence. Historical Drive notebooks move unchanged into a dated archive with checksums and original names; active root contains two workbenches. Never move datasets, experiment folders, source bundles or release artifacts during notebook cleanup.

## References

- https://otexts.com/fpp2/simple-methods.html — Naive/random-walk forecasting reference.
- https://xgboost.readthedocs.io/en/stable/gpu/ — CUDA device selection; project runtime stays pinned, not upgraded to current documentation versions.
