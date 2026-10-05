# Free-data research implementation

Current sequence: [1 October research plan](RESEARCH_PLAN_20261001.md). Saved-prediction
review and the bounded T+5 information pilot are implemented. Both workbenches now
default to read-only workflows; external-source training still requires admitted
as-published packages. The completed market ablation must not be restarted.

## Boundaries

Training is Colab-only. CT=F T+1/T+5, legacy R2 and research-v2 remain unchanged.
Data budget is zero; no paid resource is provisioned. Existing application DB and
runtime are not used for research. A modern historical split remains reused research
evidence; 2024+ audit never selects a recipe.

## Data decision for the next Colab experiment (2026-09-29)

The R2 raw and processed snapshot files were rechecked against the frozen
`corrected-v2-day1-r2/ready.json` checksums. The prepared research split has
eight 2016–2023 folds, 1,008 outer origins, and four market-derived feature
groups with 24, 39, 14 and 64 columns. This establishes a complete **market
proxy research** input using the existing Cotton, DXY and WTI series, with no
new external feature admitted. It does not certify the continuous-futures
proxy as a physical Cotton price or an as-published market-data vintage.

The USDA/ECB data work has a concrete outcome rather than an open-ended
training dependency: AMS has 999 checked daily reports for 2020–2023; NASS has
311 checked Upland condition weeks for 2010–2023; WASDE has checked historical
tables; ECB has a checksummed FX archive. They remain separate diagnostic
evidence because actual first availability, revisions or use rights are not
fully established. No score from those sources will be attributed to the next
experiment. The new `research-market-v1` namespace is for controlled Colab
feature comparison on the frozen R2 market data. A genuinely new information
source requires a separate reviewed package, cohort and experiment identity.

## Delivered entry point

As of 2026-09-29 the active entry points are
`ml/notebooks/cottonlens_data_workbench.ipynb` and
`ml/notebooks/cottonlens_research_workbench.ipynb`. The research default now
prepares a new `research-market-v1` experiment, runs its small diagnosis, then
compares four packages derived solely from the frozen R2 market data. Run-all
therefore trains in Colab and can take hours; later stages use the single
`WORKFLOW` selector. No USDA/ECB information is admitted to this experiment.
The data default is inventory, with explicit batch raw acquisition. Neither
notebook certifies unreviewed publication timestamps. See
`docs/RESEARCH_RESET_20260928.md` for evidence, limitations and the revised plan.
Eleven previous Drive notebooks, including saved Colab outputs, are preserved
with checksums under `archive/notebooks-20260928`; only the two workbenches are
active in the Drive root. The entries below document legacy launchers.

`ml/notebooks/cottonlens_free_research_colab.ipynb` loads the immutable
`sources/cottonlens-free-research-source-v1.zip` and starts a NEW
`research-free-data-v1` namespace. Existing notebooks/source bundles are preserved.

For the already prepared experiment, `ml/notebooks/cottonlens_free_research_diagnose.ipynb`
is a one-click Colab entry point for only the eight small GPU/learning controls.
It pins the existing source ZIP and ready identity, shows saved-fit progress, and
stops after diagnosis. It does not run ablation or search; the controls are not
market performance evidence. New USDA/ECB archives remain model-ineligible until
their historical publication timing is reviewed.

After diagnosis, `ml/notebooks/cottonlens_free_research_pilot.ipynb` runs only
the first base-feature XGBoost T+1 candidate on fold 1's three inner blocks.
Its six real-data fits measure runtime and verify durable resume before the
128-unit ablation. The single-candidate score is not used for model selection.

1. Mount Drive, verify/extract source, install the locked environment.
2. Supply reviewed publication packages before the first prepare (empty means only
   existing information). Changing packages requires a new experiment name.
3. Run `diagnose`, then `ablate`. Diagnostics run tabular GPU controls, not the
   previously completed scale/regularization historical sweep.
4. `search` opens 128-trial Optuna XGBoost/CatBoost search only after reviewed new
   information and complete ablation. Features are selected within each outer fold
   from that fold's past inner evidence, never from future outer results.
5. `report` does not fit models. `lock`, `reproduce`, `export` remain explicit steps.

The ablation uses 16 recipes per family/horizon/fold/group with seed confirmation,
not just 16 fits overall. Four groups, two families, two horizons and eight folds
mean 128 outer decision units; each unit contains inner fits and refits. Expect hours,
not minutes; measure the first unit before deciding session length.

## Resumption and observability

Each fit emits running/local-complete/durably-saved or cache-restored events. A fit
is durable only after payload checksums and its completion marker. Failed upload
in the SAME runtime resumes the local completed fit without retraining. If the
runtime disk was destroyed before upload, that uncommitted fit must be recomputed;
already durable fits remain reusable. Status does not claim a full payload audit.

Reports read a compact timing index rather than every fit result. Legacy missing
timings are reported as unindexed, not zero compute. Writer locks record host/PID,
start time and ownership token. Recovery requires explicit confirmation that the
old runtime stopped, no local writer, and the exact owner token; no automatic TTL
deletion. A Drive mount is not a distributed lock service: use one runtime only.

Optuna proposals and results are immutable. Trial ordinal seeds reconstruct the
same sequential sampler on resume. Pruned trials are not eligible; completed trials
contain all three inner blocks. Top five recipes use seeds 17/42/101. No Optuna DB
is hosted on Drive. The existing per-fit ledger retains payload identities.

## Free-source package workflow

`python -m cottonlens_ml.sources.public archive --kind ams --url
https://www.ams.usda.gov/mnreports/cnddsq.pdf --output <raw-directory>` archives a
bounded public response with retrieval time and checksum. Retrieval time is NOT
historical publication evidence; archived bytes are initially model-ineligible.

`python -m cottonlens_ml.sources.public compile --review <review.json> --output
<new-package-directory>` compiles reviewed release snapshots. The review contains:

- `kind`: ams, wasde, export_sales, crop_progress, cftc, pink_sheet or power.
- `features`: unique source-prefixed numeric column names.
- `usage`: cost_tl=0, research_allowed=true, terms_url, and any redistribution limits.
- `files`: relative filename to SHA256 map; includes the original source and evidence.
- `max_age_days`: explicit integer freshness policy (1–366).
- `releases`: complete feature snapshots with values, observed_through, published_at,
  vintage_id, source_url, source_file, publication_evidence_file,
  vintage_evidence_file, timestamp_verified=true.

Times require explicit timezone. Observation end cannot exceed publication time.
Features join strictly before the prediction clock and expire after max_age_days.
Per-source metadata is isolated, so multiple release packages cannot collide.
The reviewer must establish actual first-release/vintage evidence: neither a release
calendar nor the latest revised API table is sufficient. Compile validates the
contract; it cannot independently certify the truth of a human evidence review.

AMS structured historical API access needs a free personal API key. Do not paste
keys in chat, source URLs or manifests. No keys are needed for public report PDFs.
Authenticated AMS/FAS/NASS adapters archive bounded responses without exposing keys.
NASS normalization retains suppression codes and separates database load time from
publication time. See [USDA setup](USDA_COLAB_SETUP.md). Actual authenticated calls
still require the user's free keys; normalized latest data remains ineligible until reviewed.
Separately, public ESMIS NASS Crop Progress TXT and AMS daily spot-excerpt TXT
archivers need no keys. The NASS content audit now covers 311 Upland condition
weeks from 2010–2023: 308 full matches and three API-missing zero categories
flagged separately. The 2023 subset matched 23/23 weekly reports;
the official NASS calendar lists all 23 with a 4:00 p.m. ET scheduled time and
`Published` status. The AMS spot-excerpt archive now has 999 verified dated
reports for 2020–2023 and an explicitly ineligible diagnostic CSV in
`output/ams-diagnostic/`. Earlier ESMIS coverage begins in November 2018 and
contains duplicate or misdated links requiring separate review. These are
content checks, not release-timestamp certification or model-ready features. The
NASS CSV, manifest and raw-evidence ZIP are in
`output/usda-nass-history-2010-2023/` and mirrored to Drive
`reports/nass-diagnostic`. See
[vintage review](USDA_VINTAGE_REVIEW.md).

The free ECB EXR adapter archived 2010–2023 daily EUR reference-rate series for
BRL, CNY, INR and USD: 14,380 rows, including 36 explicitly missing `H` observations.
PKR is absent from this ECB reference-rate set. Rates are local currency units per
EUR, so USD crosses would be derived only after a reviewed publication policy.
`includeHistory=true` must not be treated as proof of the original release time;
these archives remain model-ineligible. The checksum audit is
`output/ecb-fx-2010-2023/archive-audit-2010-2023.json` and its private Drive copy
is under `reports/ecb-fx-2010-2023`. ECB's [reference-rate methodology](https://www.ecb.europa.eu/stats/pdf/exchange/Frameworkfortheeuroforeignexchangereferencerates.en.pdf)
and [free reuse policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html)
are the source and usage references. Quote ECB statistics when presenting the rates.

## Current evidence and remaining implementation

The saved research-v2 A report remains the baseline: every price model failed its
gate. Lower regularization alone was not supported. T+5 XGBoost direction was
55.26% against roughly 54.46% past-majority, not proof of transferable advantage.

Delivered: source preservation, owner locks, upload retry reuse, compact report
index, four feature ablations, reviewed multi-source package ingestion, source
archive/compiler, initial Optuna tabular search, 10,000-replicate research bootstrap,
compatible feature export reconstruction, notebook and no-fit regression tests.

Also implemented: replayable adaptive extensions with plateau checks; fold-local
direction, MLP/LSTM/TCN, history/cadence and nonnegative ensemble rounds; past-matured
residual interval research reports; and an explicit rebuildable MLflow projection
using a local database and immutable checksummed snapshots. These paths still need
real Colab validation. `search` selects ROUND A/B/C/E; D uses `ablate`. `track` is
explicit and does not fit models. Intervals are research outputs, not runtime/API fields.

Not yet delivered: fully populated historical USDA packages; full-history AMS
spot coverage and FAS normalizers; market/FX collectors; geospatial/text feature pipelines; TFT;
schema-v4 uncertainty/API/replay migration. These remain implementation work.

Colab GPU smoke, actual ablation/search, reproduction and release parity are still
required. No model improvement or complete-plan delivery is claimed from local tests.

## Verified availability bounds

The reviewed-source compiler also accepts an evidenced upper bound for a specific
data version. These packages use `availability_schema=verified-availability-v1`
and join on `available_at`; unknown `published_at` stays null and
`timestamp_verified` stays false. Legacy exact-publication packages retain their
existing schema and join clock.

A bounded release must name a checksummed `source-availability-upper-bound-v1`
receipt. The receipt binds `available_by`, `source_sha256`, `vintage_id`, reviewed
timing/version flags, and checksummed underlying `evidence_files`. Accepted bases
are `official_release_record` and `contemporaneous_archive_capture`. A reviewer
must verify those assertions against the evidence; validation is not automatic
certification of historical truth. Schedules, embargoes, arbitrary lags and a
present-day download cannot establish historical availability. Freshness is
measured from the evidenced availability clock. Equal-time origins cannot use a
new release. This contract does not admit any existing unverified archive.
