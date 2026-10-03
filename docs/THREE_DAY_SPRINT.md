# Three-day tabular sprint

## Execution and preservation

Use `ml/notebooks/cottonlens_sprint_colab.ipynb`, T4 Colab, experiment
`tabular-sprint-v1`. The source delivery is `cottonlens-sprint-source-v1.zip`
and its SHA-256 sidecar in Drive's `CottonLensAI/sources` directory.
The notebook is also delivered directly to the synced Drive folder.

Run code cells 1–6 for the first search. Cell 6 has `STAGE = 'search'`.
Then change only that value, running one stage at a time:
`refine`, `cadence`, `lock`, `benchmark`, `reproduce`, `release`.
The final notebook cell validates the exact pending release and generates the
shareable report. Do not run that cell before `release` completes.

The smoke runs only once per exact source; restarting the notebook reuses its
successful immutable report. A failed smoke must be inspected, not bypassed.
The 48-hour search deadline starts when sprint readiness is first frozen.
After the deadline, `lock` can use completed initial recipes; missing recipes
remain missing rather than being scored as failures or successes. `refine`
requires all six initial recipes per horizon. No new research is allowed after
`lock`; benchmark scores cannot reopen search. A stale writer lock requires
confirmation that the old Colab process stopped before manual removal.

R2 source, raw/processed snapshots, readiness, 504 benchmark origins, releases,
and checkpoints remain unchanged. New results, attempt ledger, checkpoints and
releases live under `experiments/tabular-sprint-v1`. Only its own release pointer
is updated. Cached model reuse requires the matching data/code/environment
identity and checksum-complete checkpoint. Fresh reproduction ignores model
caches and writes separately. A completed reproduction receipt may be reused.

## Frozen research decisions

Development: last 126 common full-feature, label-valid observations at each
2018–2021 year end; no T+5 target reaches 2022. Snapshot inspection confirmed:

| Fold | Start | End | Scored origins | Purged training rows / validation rows |
|---|---|---|---:|---:|
| 1 | 2018-06-18 | 2018-12-31 | 126 | 1877 / 126 |
| 2 | 2019-06-14 | 2019-12-31 | 126 | 2114 / 126 |
| 3 | 2020-06-09 | 2020-12-31 | 126 | 2341 / 126 |
| 4 | 2021-06-09 | 2021-12-23 | 126 | 2568 / 126 |

Initial XGBoost recipes: depths 1/2/3 crossed with squared-error or
training-price-weighted absolute-error on log returns. Fixed learning rate .03,
subsample/colsample 1, min child weight 20, lambda 10, alpha 1, seed 42,
1200-tree ceiling and 50-round price-MAE early stopping. Price-weighted
log-return L1 is not an exact price-MAE objective. Every fit/refit recomputes
weights using only its own past fitting rows.

Conditional refinement: initial best must have positive aggregate gain and
three fold wins. Exactly three additional variants: five-year history,
volumeless Cotton, or volumeless Cotton plus macro. No additional grid. Exact
score ties prefer shallow models, then fewer active features, then recipe hash.
Cadence: one comparison of 126-origin static blocks versus 21 source Cotton
observations; 21 wins only with >=1% lower aggregate MAE, >=3 fold wins versus
126 and nonworse direction. Otherwise 126 remains. 21 is approximately monthly,
not a verified optimum. No tuning occurs at intermediate refits.

Historical targets mature strictly before each fitting cutoff. For cadence 21,
the source observation ordinal advances even on unscored feature-missing rows.
Scored origins remain identical. Natural feature availability affects fitting
coverage, never replacement of frozen evaluation dates.

Naive, Ridge, past median-return and past majority-direction are references.
The direction-only reference uses a tiny signed return and is not a production
candidate. Only XGBoost can replace Naive in this sprint. The existing 5% MAE,
53%/55% direction and 3-of-4 gates remain unchanged. Previously seen 2024+ audit
is accessed only after both horizon decisions and fresh reproduction, purely
descriptively; it cannot tune, select or veto.

## Runtime and evidence

Feature subsets keep the existing ordered 24-column public schema. Training
sets inactive columns to zero; native trees must never split on them. The
ordered active list is in the booster attribute and recipe identity. Raw full
input inference and TreeSHAP equivalence are tested in the Colab synthetic
smoke before market training. This avoids adding a custom runtime dependency
or changing artifact schema. Forecasts keep log-return semantics.

Benchmark and reproduction preserve per-origin predictions and refit cutoffs.
Reproduction requires fresh fits and maximum prediction difference <=1e-6.
The release uses the last benchmark block's locked tree count for static seen
audit and a separate current deployment refit. Both roles have distinct model
identities. No current-deployment score is presented as historical evidence.

## Presentation fallback and five-minute narrative

R2 is installed in `output/presentation/r2-demo`, with an isolated
`presentation.db` and artifact directory. It is measured evidence, not fixture
data. Start from the repository root:

```powershell
python backend/scripts/presentation.py serve --name r2-demo --port 8080
```

Open `http://127.0.0.1:8080`. The built frontend is served by the same local
process. Real R2 data can be browsed without external network access.
Installing a later approved sprint release uses a **new** presentation name:

```powershell
python backend/scripts/presentation.py install --name sprint-demo --bundle PATH_TO_VALIDATED_ZIP
python backend/scripts/presentation.py serve --name sprint-demo --port 8080
```

Never point these commands at the original database/runtime. Its unresolved
historical DB recovery incident is not repaired by creating this demo.

Five-minute narrative: 45 seconds for the Cotton proxy and T+1/T+5 question;
60 seconds for timing/purge and frozen comparisons; 90 seconds for the actual
model table and Naive selection; 60 seconds for Forecast/Why/Sensitivity/Replay;
45 seconds for Colab-to-lightweight-runtime reproducibility and limitations.
Say “60 retained complete-feature observations” for R2 LSTM, not 60 sessions.
Sensitivity is experimental when Naive is primary and is never causal evidence.
If sprint gates fail, show that outcome alongside R2 rather than hiding it.

R2 result card: Naive MAE 1.502 / 3.626 cents per lb; XGBoost MAE 1.503 / 3.634,
direction 50.60% / 48.61%; LSTM MAE 1.520 / 3.703, direction 47.62% / 45.44%.
These are seen historical-development scores on 504 common origins. No
independent future advantage has been established. Final sprint scores remain
pending Colab execution; no improvement is claimed by these code changes.

## Local verification before source delivery

The ML suite passed 112 tests; after adding fold/reference/coverage reporting,
the 15 sprint tests passed again. Backend passed 25 tests against temporary
databases. Python lint and whitespace checks passed. Node 24.19 completed
`npm ci` and the production build. The five demo pages were checked at 1440,
1024 and 390 px with zero horizontal overflow while external network requests
were blocked; screenshots are retained under `output/playwright`.

The original DB, preservation ZIP and R2 source ZIP retain their recorded
SHA-256 hashes. Historical DB recovery remains unverified. No market-data
model fit ran locally. Colab environment smoke, real trial results, fresh
reproduction and the new release's runtime parity remain execution gates;
passing local tests does not claim those gates have passed.
