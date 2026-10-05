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

### Shared-model raw diagnosis completed

288 signed inner fit receipts reconstructed all 32 yearly selected scores/weights within 1e-12; 355 input-evidence hashes were verified with zero fits. Repeated inner dates remain dependent. Raw pooled forecasts gained 0.1985% / 0.9126% against the raw Cotton-only control, winning 8/8 / 7/8 years, but raw Naive gains were only +0.0239% / -0.0990%. The exploratory T+5 raw matched-control interval excludes zero; both raw-versus-Naive intervals include zero. This post-hoc comparison cannot replace the preregistered selected-strategy gate or establish release superiority.

Pooled outside prediction standard deviation is 59% / 58% of the control's, and their predictions correlate 0.835 / 0.896. This is compatible with reduced noise; it does not prove a causal denoising mechanism or stronger information. Outside rank IC is small (0.0252 / 0.0339), raw direction 50.15% / 51.89% fails 53% / 55%, and raw T+5 log-return R² versus zero is negative. Removing shrinkage is unsupported.

Selected inner scores average equal-weight relative block errors after past-only weight selection; raw aggregate metrics pool price errors over all prediction slots. They are different measures, not interchangeable success criteria. Realized-movement bins remain explanatory and never remove origins.

The fixed shared-linear hypothesis is closed as insufficient for the price gate. Preserve its exploratory matched-control improvement; no larger mixture/alpha/cadence search is opened. Absolute-error objectives already exist in the earlier tabular recipe generator, so an objective change cannot be called untried without evidence of actual executed trials. Next bounded task: inventory completed model/objective/feature combinations, distinguishing executed receipts from available recipes, then preregister one distinct hypothesis. Evidence: `output/reports/agri-transfer-diagnosis-20261003/completion.md`. Frozen ML source, existing fits, DB and D21 are unchanged; 12 diagnostic files were checksum-readback verified in the Drive directory.

The existing Experiment/Ledger and two workbenches are used. New metadata catalogues allow status/compare to restore preparation and result batches without scanning model packages. Resume verifies completed work; incomplete or corrupt checkpoints cannot silently become completed experiments. Run All defaults to read-only status. GPU/sequence training stays in Colab. The isolated CPU group does not install TensorFlow/CUDA into the backend.

## Evidence and operating limits

### Side-conversation follow-up queue — 2026-10-03

The bounded completed-combination inventory now records 722 decisions and 8068 candidate
evaluations (`output/reports/completed-combination-inventory-20261003/v2/inventory.json`).
These repeat across years/groups and are not unique fits or independent evidence; full
checkpoint payload availability was not reverified by this inventory. Squared and absolute
objectives already have executed legacy XGBoost/CatBoost records.

The main conversation is preparing `agri-nonlinear-pilot-v1`; this side-conversation update
does not interrupt that work or claim training/completion. Follow-up order: a no-fit,
same-origin MAE/MAPE/OOS-R²/Naive bridge table; finish the active fixed nonlinear comparison;
then consider a bounded ARIMA baseline registration and, only with a distinct hypothesis,
a small matched LightGBM pilot. These are queued proposals, not implemented runners or
automatic training authorization. No broad search or simultaneous new architecture sweep
is opened. DeepAR/N-BEATS remain conditional; the existing DLinear/exogenous ordering,
two workbenches, single ledger, gates and D21 remain unchanged. See the final
"Yan sohbet incelemesi" section in the canonical plan for scope, dependencies and decisions.

- Price gate: at least 5% MAE improvement, T+1 53% / T+5 55% all-origin direction accuracy, at least 6/8 yearly wins. Flat forecasts are reported; conditional nonflat accuracy never replaces this gate.
- Paired bootstrap: 10000 replicates, fold boundaries preserved, main block length 20 and sensitivity lengths 10/40. Post-hoc power assumptions and reused-history selection limits remain explicit.
- The Windows collector records prospective Naive/EWMA forecasts and source vintages with cutoff/missing/no-backfill rules. Computer uptime and local Drive copy verification do not establish continuous service or cloud synchronization.
- New sources require specific availability/vintage evidence for Tier B. Public access alone does not establish redistribution rights. Data budget is zero; paid resources need separate approval.
- Old databases, preservation backups, snapshot/release identities and the 24-column production artifact remain protected. The 83-column challenger cannot enter runtime without exporter/importer/runtime/parity contracts being upgraded together.
- All ML tests passed locally in the isolated CPU environment: 472 passed, one skipped. Changed Python files pass Ruff. Prior GitHub backend, frontend, ML data contracts and Compose checks passed; the latest commit's CI status must be checked separately. This does not verify GPU training or Linux Colab execution. A prior broad Ruff scan reported six import-order findings in unchanged test files.

Canonical decisions: [MASTER_PLAN_20261001.md](MASTER_PLAN_20261001.md). D21 remains 22 October 2026 at 20:06:59 UTC; adding a new experiment does not restart that date. Local data, model files, credentials and private handoff material are excluded from Git.


## Fixed nonlinear shared-training pilot: implementation ready, results pending

Completed-combination inventory reviewed722 signed decisions/8068 candidate evaluations
and48 sampled markers; it did not reverify all old checkpoint payloads. The pooled
agricultural comparison previously used Ridge only in this scoped inventory.
One distinct fixed XGBoost comparison is now implemented as `agri-nonlinear-pilot-v1`,
experiment `research-agri-nonlinear-pilot-v1`: Cotton-only versus shared Cotton/corn/soybean,
same six causal features/2006 origins per horizon, Cotton-fit transforms and matched total
sample weights. CUDA/hist only, no extra parameter/feature search;772 maximum fits/32 outputs.
Cotton-only past validation controls early stopping; yearly median tree counts propagate
through21-observation refits. Completed fits and metadata remain on the existing ledger/mirror.

Local ML verification:477 passed/one skipped; changed Python files pass Ruff.
GPU estimators were mocked, not trained locally. Actual Linux/CUDA execution and any new
market result remain pending. Tier A blocks release regardless of numeric gate results.
The existing Drive Research Workbench is delivered with this profile, status/RUN_TRAINING=False;
choose pilot deliberately in a T4/L4 session, initially MAX_MINUTES=15, and resume unchanged.
The old frozen bundles/experiments and DB are retained. D21 is unchanged.
Completion: `output/reports/agri-nonlinear-implementation-20261003/completion.md`.


## Fixed nonlinear Colab outcome: complete, gates failed

772 distinct saved fit IDs in the completed console log;32 annual outputs and34 small
metadata packages verified.32 checkpoint spot-checks confirm NVIDIA L4/cuda:0, fixed
asset weights, label maturity and sampled annual tree-count propagation;96 payload
hashes checked. This is not full772-payload reproduction or deployment parity.
Cotton-only Naive MAE gain:+0.038311%/+0.213315%; pooled:-0.051932%/-0.035006%.
Pooled inner contribution:+0.015967%/+0.373482%,3/8 and4/8 inner wins. Both price gates
and priority signals fail. Raw forecasts also lose Naive; no shrinkage removal or expanded
shared-model search. Flat forecasts explain low all-origin direction percentages; report
the full cohort rather than replacing the gate with conditional nonflat accuracy.
Same-origin MAE/MAPE/actual OOS R² bridge is recorded. Naive remains primary/Tier A blocks
release; D21 unchanged. The existing Drive workbench returns to status/RUN_TRAINING=False;
no repeat GPU run is needed. Next bounded work: preregister small statistical-reference
feasibility without assuming ARIMA local CPU permission. Review evidence:
`output/reports/agri-nonlinear-colab-review-20261003/completion.md`. Zero new fits.


## Statistical-reference pilot prepared; market results pending

The existing matched runner now supports a small past-only candidate selection:
Naive, unshrunk training-mean drift reference, ARIMA(1,1,0)/(0,1,1). Same2006 origins per
horizon, annual past-only selection/shrinkage,5-observation purge and21 refit cadence.
ARIMA coefficients use mature purged training prices; observed-only state filtering
updates inputs through each origin without re-estimation. No auto order/seasonal search.
statsmodels0.14.5 is already locked in the separate CPU group. Colab CPU-only permission
is explicit; local CPU allowlist remains unchanged. At most964 ledger jobs/482 numerical
ARIMA estimations/32 annual outputs. Invalid inner candidates are signed/excluded;
selected outside failures block instead of replacing forecasts after seeing outcomes.
483 local ML tests passed/one skipped; after a helper-hash check, five focused tests passed.
Only synthetic/mocked fits ran locally; no new market-data model/score or reproduction
claim. CI tests the same installed statsmodels API synthetically. The existing Drive
Research Workbench is configured for statistical-pilot-v1, status/RUN_TRAINING=False,
initial15-minute session; CPU runtime suffices. Old source packages/data/DB remain intact.
Canonical gates, no-selection2024+ audit and D21 are unchanged; automatic release closed.
Evidence:output/reports/statistical-implementation-20261003/completion.md.


## Statistical Colab outcome: complete; close the fixed order pilot

32/32 annual outputs verified on the same2006 Cotton origins/horizon.41 metadata
packages restored;932 distinct saved computation IDs in the console log (including
analytical references, not932 ARIMA fits). Six numerical inner candidates were excluded.
Past-selected Naive/drift/ARIMA strategy Naive MAE gains:-0.287484%/-0.087902%; direction
36.79%/23.73%, year wins1/8 for each horizon. Both gates and research-priority fail.
Raw strategy also loses Naive. Naive/flat choices explain low all-origin direction;
do not condition on nonflat forecasts or remove shrinkage.2017 T+1 inner score0.986502
transported to a2.919244% outside MAE loss. Original data/year cohort remains unchanged.
Saved bootstrap report replayed within1e-12 numeric tolerance; no new fit, full payload
audit, fresh reproduction or runtime parity. Naive stays primary; no automatic wider
ARIMA/LightGBM/sequence search. Next: consolidate completed evidence and register one
distinct bounded information/model hypothesis. D21 and2024+ selection prohibition
unchanged. Existing launcher returns to status/RUN_TRAINING=False; old DB preserved.
Evidence:output/reports/statistical-colab-review-20261003/completion.md.


## Bounded nonlinear temporal-representation contrast prepared — 3 October 2026

Four completed matched pilots were consolidated:128 signed annual outputs,2006 exact
common origins/prices/targets per horizon; no selected strategy meets the price gate.
Scoped completed inventory has no XGBoost83-column dense-return-path comparison. This
does not prove that no unavailable legacy experiment tried it. Linear60-observation
Ridge path and fixed nonlinear six-feature transfer tested separate representations.
One distinct mechanism test is nonlinear interactions in the same recorded path;
it adds no new information channel and is not a broad GPU/hyperparameter search.

nonlinear-path-pilot-v1 / research-nonlinear-path-pilot-v1: fixed CUDA XGBoost depth2,
eta0.03,child weight20,L1=0,L2=1,seed42,sampling1,scaled log target,squared loss;
base24 versus base+59 preceding returns83. Max600 trees/patience50, three past63-origin
inner blocks; yearly median tree count and shrinkage locked before outside results,
21-observation refits,5-observation purge/mature labels. Both arms use the same frozen
2016–2023 origins; no row dropping for missing features or time compression.
Use the preserved original24-feature snapshot, not the six-column transfer snapshot.
Terminal2024+ targets are masked only in a new packet; old snapshots untouched.
Budget772=96 stopping fits+676 refits,32 annual outputs; no new windows/recipes/seeds.
P1/P5 paired price-MAE,10000 bootstrap,20 main/10/40 sensitivity,BH; reused-history
evidence. Same research-priority and5%/53%/55%/6-of-8 price gates; no automatic release.
Negative result closes this fixed hypothesis without expanding the same path grid.

Existing Experiment/Ledger/matched runner and two workbenches retained. Colab GPU-only
fitting; CPU fallback error. Status/compare use CPU; defaultstatus/RUN_TRAINING=False.
489 ML tests passed/one skipped; final frozen-packet preparation test also passed.
Only mocked/synthetic fits locally; no new market score or actual CUDA/reproduction
claim. D21,2024+ no-selection audit, source availability limits,old DB and backups intact.
Evidence:output/reports/research-decision-20261003/evidence.json;
preregistration/implementation:output/reports/nonlinear-path-implementation-20261003/.

Frozen design:32ba7324583d22918562bfb9ed575d2a562224ca56e84afa68cccfdc7c1c5dd8.


## Fixed nonlinear return-path Colab outcome: complete; hypothesis closed

772 distinct saved fit IDs,32 outputs and34 metadata packages verified. Eight checkpoint
spot-checks confirm NVIDIA L4/cuda:0, payload hashes/mature labels and sampled locked
annual tree counts; not a full772-model audit, reproduction or deployment parity.
83-column Naive MAE gains:-0.004520%/-0.538577%; year wins3/8 /0/8.24-column control
gains:-0.105540%/-0.473757%. Inner path gains:-0.101730%/-0.084072%,5/8 /2/8 inner wins.
Both priority/price gates fail; BH P1/P5 p0.541146/0.692231. Raw path forecasts also lose
Naive(-0.120841%/-1.163883%); removing shrinkage is unsupported. Flat forecasts504/1002
explain selected all-origin direction36.89%/24.58%; do not replace the gate by nonflat
coverage. Close fixed nonlinear path, no wider lag/depth/seed/history search or release.
Naive primary,2024+ selection prohibition/D21 unchanged. Existing launcher status/
RUN_TRAINING=False; GPU may close. Zero new fits; old DB and unrelated working tree kept.
Evidence:output/reports/nonlinear-path-colab-review-20261003/completion.md.

## Unified session and matched evidence update

The existing Research Workbench now continues planned checkpoint pauses automatically
within SESSION_MINUTES (default240; CPU maximum240, GPU maximum720). Frozen runners
still receive segments no longer than60 minutes; fit code, sources and identities are
unchanged. Complete cached stages need no new fits. Failure, interruption, missing
terminal JSON or a pause without new saved progress stops the session; no lock removal
or automatic failure retry. The budget is soft at fit/checkpoint boundaries. Compare
runs once after completion, not after each paused segment. Run All remains read-only.

40 scoped orchestration/notebook/supervisor tests passed; Ruff and diff checks passed.
No actual model training or GPU throughput claim. Five matched pilots now contribute
160 signed annual outputs on the same2006 origins per horizon; all selected price
gates fail. Evidence:output/reports/research-decision-20261003/v2/evidence.json.
Original consolidation and old release/snapshot/DB files remain intact.

Decision: do not restart the completed pilots or open another same-feature parameter
grid. A further fit requires a distinct registered information/model hypothesis and
a finite budget. Session automation does not authorize a queue of new experiments or
change the22 October decision checkpoint, price gates or seen-audit restrictions.

## WASDE narrative source preflight: complete, not training-ready

Added offline cotton-narrative extraction and causal assumed-date alignment in
sources/wasde_text.py. Existing TXT files are tables; PDF commentary is a distinct
candidate information channel.22 already-listed public PDFs acquired at zero cost;
old versions preserved. Final signed corpus-v4 contains90 distinct reports/narratives,
2016–2023, about219 words/report. Eight dated entries are explicitly excluded for
date/layout/version ambiguity. URL aliases follow existing archive provenance rules;
full URL matching avoids confusing multiple latest.pdf files. Exact parser snapshot
and source/receipt/PDF hashes retained. No publication/vintage admission inferred.

On the preserved1254-origin2019–2023 WASDE cohort, lag1/2 have no missing text;
lag6 has five missing origins, retained. At least24 distinct past documents before
every tested inner cutoff (availability preflight, not final target-maturity test).
42 focused WASDE/parser/alignment tests passed; Ruff and diff checks passed. No fits.

Next proposal: one fixed Ridge/TF-IDF128/SVD8 incremental-text comparison, T+5 only,
against age/missingness and numeric-balance controls. Train-only unique documents;
nominal lag1 with additional1/5-observation stresses. Planned45 annual outputs and
954 refits, no parameter sweep. This remains a proposal until training adapter,
checkpoint identities and workbench integration pass tests and exact recipes freeze.
Five-year Tier-A discovery cannot establish the eight-year release gate or independent
success; Naive stays primary. Existing launcher remains status/trainingFalse.
Evidence:output/reports/wasde-text-audit-20261003/hypothesis-v2.json.

## WASDE narrative pilot implementation: ready for a bounded Colab run

The existing Experiment/Ledger and Research Workbench now support wasde-text-pilot-v1;
no third notebook or parallel trainer. Three arms at each lag: base24 plus text timing
(26 inputs), numeric WASDE plus the same timing (31), and numeric plus TF-IDF/SVD text
(39). One Ridge alpha1, scaled-log target, seed42, T+5 only. Vocabulary/IDF/SVD learn
only unique eligible training documents; daily repeats cannot inflate document counts.
Missing text remains an origin with imputation/indicators. Exact vocabulary, IDF,
components, numeric preprocessing, training document IDs and column order are saved
in the fit payload and covered by the existing ledger checksums.

Original1254-origin2019–2023 split,21-observation refit, five-observation label maturity,
three63 past inner blocks and shrinkage grid unchanged. Lag1/2/6 arms isolate timing
assumptions. Finite budget954 refits,45 annual outputs; no wider parameter search.
Prepare checks mature-document availability without fitting. Cached decisions/outputs
are checked against frozen recipes/origins/labels; missing or corrupt evidence stops
without silent retraining. Metadata catalogues restore completed reports separately
from model payloads. Auto-session/Run All trainingFalse defaults remain.

ML suite:547 passed,1 skipped; synthetic fixtures only. Source/parser, workbench,
ledger/metadata restore and transform round-trip contracts verified. Real narrative
pilot training, CPU timing and model performance remain pending in Colab. No release,
eight-block gate or Cotcast comparison is claimed. Tier A and D21 unchanged. Old DB,
releases, source bundles and preservation backups untouched; no commit/push.
Inputs and execution receipt:output/reports/wasde-text-pilot-implementation-20261003/.

## WASDE narrative Colab outcome: complete; fixed hypothesis closed,4 October2026

Reported16.02 minutes,45/45 verified annual outputs,1254 frozen origins. Restored three
metadata packages and checked outputs against recipes, decisions, dates/prices/labels.
No local fits.954 registered refits is a budget, not a full payload audit: cold Drive
payload scan stopped; no post-run reproduction/deployment parity claim.
Text lag1/2/6 Naive gains:-0.210299% /0% /-0.755777%, all0/5 strict yearly wins.
Lag2 shrinkage selects zero every year (exact Naive). Lag1 inner contribution vs numeric
is-0.309442%,0/5; outside contribution only+0.049843%,1/5. All priority tests fail.
Raw text forecasts lose Naive by10.85–11.72%; removing shrinkage is unsupported.
Close this fixed Ridge/TF-IDF/SVD hypothesis; no automatic encoder/seed/GPU expansion.
Tier-A/five-year research does not evaluate the eight-block release gate. Naive primary,
2024+ selection prohibition, original gates and22 October checkpoint unchanged.
DB/source/old results preserved. Evidence:output/reports/wasde-text-colab-review-20261004/.

## Next registered hypothesis: joint fundamental state,4 October2026

Profile/experiment:fundamental-joint-pilot-v1/research-fundamental-joint-pilot-v1.
Reuse frozen FAS/NASS/Texas-weather sources together; reviewed runs used them
separately. Preserve2006 T+5 origins over2016–2023 and all missing rows. Same past-only
selection/maturity/refit protocol; three lag stresses1/2/6. Complete interaction
coverage886/2006 at each lag is reported separately, never used to select origins.
Three arms: timing controls31 inputs, joint numerical sources46, fixed crop-condition
products49. One CPU Ridge alpha1/seed42/scaled-log plus existing shrinkage per arm.
Budget1521 refits(648 inner/873 outer),72 annual outputs. No parameter search.
Compare interaction with BOTH controls at all lags; research priority requires
>=0.5% inner gain,>=5/8 inner wins,positive outer gain and>=5/8 outer wins.
No release/source admission/automatic larger search. Publication/vintage remains
unverified Tier A. Original gates,Naive primary,2024+ prohibition and D21 unchanged.
Implementation/preflight evidence:output/reports/fundamental-joint-implementation-20261004/.
Real pilot fitting and predictive results are pending Colab; tests use synthetic data.


## Availability-clock correction and completed pilot — 5 October 2026

The approved `availability-clock-pilot-v1` completed 676 CPU Ridge fits, 32 annual
outputs and 8,024 prediction rows on matched 2016–2023 origins. This is a Yahoo
availability-assumption sensitivity experiment, not verified publication timing.
T+1 selected control/available Naive gains: -0.328473% / -0.153805%; T+5:
-0.799971% / -1.339565%. Both available arms win 0/8 years against Naive.
T+1 improves slightly against control, but 87.54% of the loss reduction comes
from selecting zero weight in 2020. Raw available predictions are worse than
raw control overall. T+1 Naive-gain 95% upper bounds are 0.125691% (block20)
and 0.019337% (block60), far below the unchanged 5% practical gate.
Decision: retain Naive; no automatic grid/source/GPU expansion of this timing recipe.

Legacy LSTM's 498-versus-439 origin comparison is invalid: reconstructed matched
T+1 aggregate gain is -1.696141%, not +2.764668%. Full LSTM predictions are absent,
so no paired CI is claimed. Historical TCN known-signal gain -23.593133% failed
the learning control despite its old overall `passed`; all executed families
now require >=50% known-signal gain and cached controls are re-evaluated.
Old evidence remains immutable; corrections and 109-entry inventory are separate.
Synthetic controls are not market results; incomplete runs are not negatives;
T+5 source evidence does not settle T+1. Raw direction, selected direction and
active rate differ; zero-shrinkage [0,0] intervals do not establish no source signal.
Previously reviewed 2016–2023 and 2024+ history is not an independent holdout.

Details and exact evidence paths: [availability-clock result](AVAILABILITY_CLOCK_RESULT_20261005.md).
Existing release gates and the prohibition on 2024+ selection remain unchanged.
