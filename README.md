# CottonLens AI

Explainable Cotton No. 2 forecasting and market-sensitivity dashboard. CottonLens separates expensive model development from the lightweight interview demo:

- Google Colab performs ingestion, feature generation, Ridge/XGBoost/LSTM training, MLflow tracking, rolling-origin evaluation, explanations, and artifact export.
- The application stack serves Angular, FastAPI, PostgreSQL, and CPU inference from a verified artifact. An explicitly authorized, isolated CPU research environment can run the small full-year pilot with one job and at most two threads; it is separate from backend dependencies and CI.

The repository includes an explicitly labelled development fixture so the complete product flow can be reviewed before a real Colab artifact exists. Fixture values are never presented as trained results.

The artifact currently installed at `runtime/artifacts/current` was created with the **previous** evaluation protocol. The new walk-forward pipeline and Model Lab evidence become measured results only after a fresh Colab Run All and validated artifact import. This repository change alone does not improve an already-trained model or establish a new accuracy score.

## Architecture

Free-data research continuation: [workflow and current limitations](docs/FREE_DATA_RESEARCH.md),
[free USDA keys and Colab Secrets setup](docs/USDA_COLAB_SETUP.md),
and [new research notebook](ml/notebooks/cottonlens_free_research_colab.ipynb).
Existing research-v2 checkpoints remain a separate reference.

The current research contract is [MASTER_PLAN_20261001.md](docs/MASTER_PLAN_20261001.md).
Measured outcomes, limits and the current matched Ridge comparison are summarized
in [research status](docs/RESEARCH_STATUS_20261003.md). Completed full-year and
recency experiments did not pass the fixed price gates; Naive remains primary.
Use the existing [research workbench](ml/notebooks/cottonlens_research_workbench.ipynb)
and [data workbench](ml/notebooks/cottonlens_data_workbench.ipynb); Run All defaults
to status/readiness and does not train. The local CPU launcher is
`python ml/full_year_cpu.py setup`, followed by `status` or `compare`.
An actual pilot/reproduction requires an explicit stage, a matching frozen source
and environment, and a bounded session. Old experiments and releases are immutable.
Live archive scripts are in `ml/scripts/`; a successful task registration and local
credentials are required before claiming unattended collection is active.

```text
Google Colab + GPU                    Local Docker
┌─────────────────────────┐          ┌───────────────────────┐
│ Yahoo/CFTC ingestion    │          │ Angular 22 + Nginx    │
│ leakage-safe features   │  ZIP     │          ↓            │
│ XGBoost + LSTM          ├─────────▶│ FastAPI inference     │
│ MLflow + walk-forward   │ verified │     ↙          ↘      │
│ SHAP + artifact export  │          │ PostgreSQL   XGB/ONNX │
└───────────┬─────────────┘          └───────────────────────┘
            ▼
       Google Drive
```

TensorFlow, MLflow, Jupyter and CUDA are intentionally absent from the backend image.

## Quick start: development fixture

Prerequisites: Docker Desktop with its Linux engine running.

```bash
docker compose up --build
```

Open:

- Product: <http://localhost:8080>
- OpenAPI: <http://localhost:8080/api/v1/openapi.json>

The yellow “Development fixture” banner means the interface is using deterministic illustrative data. It is suitable for UI/API development, not a market-performance claim.

Stop the stack without removing its database:

```bash
docker compose down
```

## Train in Google Colab

1. From this working tree, run `python ml/source_bundle.py --output output/cottonlens-day1-source.zip`. Keep the generated `.zip` and `.zip.sha256` together; the bundle includes uncommitted source changes and records the Git HEAD separately.
2. Place both files in `MyDrive/CottonLensAI/sources/`, and open `ml/notebooks/cottonlens_colab.ipynb` from this same working tree in a T4 GPU Colab runtime.
3. Run cells through **preflight**, inspect the frozen `ready.json` and 504-origin `cohort.json`, then set `RUN_TRAINING = True` in the training cell only when beginning the Colab experiment. Finally run artifact validation. No repository clone, pull, commit or push is part of the notebook.

There is **one notebook with eight code cells**:

```text
Drive mount → checksum-verified local source bundle → isolated Python + locked dependencies
→ mandatory GPU/synthetic smoke → cached source validation → frozen preflight → walk-forward training
→ ZIP/checksum/backend-runtime validation → shareable results report
```

The Colab kernel can remain Python 3.13. `ml/colab_setup.py` installs uv 0.12.0 with `pip --target` into a separate tool directory (no global installation or `ensurepip` requirement), provisions managed **Python 3.12.11**, and runs `uv sync --locked --extra cuda` into `/content/cottonlens-py312`. It never installs training dependencies into Colab's global Python. All training subprocesses force `MPLBACKEND=Agg`, disable user-site imports, and stream stdout and stderr, including original tracebacks. A failed stage stops Run All before training/export can continue.

`ml/uv.lock` freezes the complete dependency graph (including CUDA wheels); `ml/constraints/colab-py312.txt` mirrors the direct pins. Do not run an unbounded `pip install -U` in either environment. Direct versions are:

```text
numpy==2.1.3             pyarrow==21.0.0          tensorflow==2.20.0
keras==3.10.0            protobuf==5.29.6         onnx==1.17.0
tf2onnx==1.17.0          onnxruntime==1.22.1      mlflow==3.16.1
xgboost==3.0.2           pandas==2.2.3           scikit-learn==1.6.1
shap==0.47.2             matplotlib==3.10.0       ml-dtypes==0.5.1
joblib==1.4.2            requests==2.32.3         yfinance==0.2.54
pytest==8.3.5
```

Build backend: hatchling 1.27.0. TensorFlow's `and-cuda` extra is installed only in Colab. MLflow 3.16.1 resolves with PyArrow 21; the older MLflow 3.1.1 requirement did not.

**Verification boundary:** the lock resolves for Python 3.12/Linux and the lightweight regression tests run locally without training. The actual GPU/TensorFlow/Keras/ONNX combination must pass the notebook's synthetic smoke in Colab; dependency resolution alone is not runtime compatibility evidence. No successful Colab run is claimed by this README.

Before any real dataset training, the smoke trains a tiny synthetic LSTM, saves/reloads it, exports both horizons, checks raw-input parity below `1e-4`, and exercises XGBoost save/load, nested MLflow runs with Drive restore, Parquet round-trip, and non-finite feature tests. It also loads the repository's actual backend runtime in a second inference-only Python environment which rejects TensorFlow, Keras, tf2onnx and MLflow. Results and exact versions are written to `reports/environment-smoke.json` on Drive. A synthetic pass is not evidence of market accuracy.

The notebook mounts Drive and writes to:

```text
MyDrive/CottonLensAI/
├── data/raw
├── data/processed
├── mlruns
├── tracking
│   ├── mlflow.sqlite
│   └── mlflow.previous.sqlite
├── reports
├── checkpoints
└── artifacts/releases
```

MLflow uses a **local SQLite database**, not a database opened directly on the Drive mount. SQLite's backup API creates consistent snapshots on Drive every 30 seconds, after runs and at shutdown. Reconnecting restores the snapshot before opening the same experiment; artifacts and training checkpoints remain on Drive. Abrupt runtime loss can lose metadata since the last completed snapshot. Legacy file-store `mlruns` records are left untouched, not silently migrated.

Only one Colab writer may use the same Drive root. If a killed runtime leaves `tracking/.writer-lock`, first stop the old runtime, then remove that empty lock directory in Drive and rerun. Do not remove a live writer's lock. Corrupt snapshots are rejected; `mlflow.previous.sqlite` is retained for explicit recovery. If final Drive backup fails, the error prints a local recovery database path: keep the Colab session open and copy that database to safe storage before disconnecting.

Invalid price observations are recorded with series/date/field/value in `data/processed/data_quality.json`. Zero, non-finite and non-positive prices are excluded from logarithms, not hidden with warning suppression. Cotton gaps are not filled or removed before target alignment; external features forward-fill only from the past and DXY/WTI are delayed one Cotton session. Current UTC-day candles are excluded because they may be incomplete. Genuine negative WTI prices are retained in raw data but excluded from log-price calculations. The report records each source's first date and the first modeling date, so the actual reason for a 2016 start can be inspected rather than guessed.

The **next Colab run** uses four frozen 126-Cotton-observation rolling-origin folds ending before 18 June 2024. Their exact 504 origins and data/code identities are frozen before model fitting. Every fold has an earlier training/inner-validation period, a five-observation target purge at both boundaries, and train-only transformations. Naive persistence and a fixed Ridge reference appear beside the learned models. A learned model must improve aggregate price MAE by at least 5% versus Naive, reach directional accuracy of 53% (T+1) or 55% (T+5), and beat Naive in at least 3/4 periods. LSTM displaces XGBoost only with another 5% MAE improvement and no directional-accuracy loss. The 2024 onward interval is a **previously observed historical audit**: it has no selection, veto or tuning role. No future accuracy is promised.

The Colab experiment budget is eight XGBoost hyperparameter configurations per horizon, four LSTM unit/dropout configurations, and fixed-configuration feature ablations. XGBoost has at most 1,200 trees with 50-round early stopping; LSTM has at most 100 epochs with 10-epoch early stopping. Protocol v2 uses a fixed 126-observation inner validation period, selects settings there, and **refits on the available pre-evaluation training + validation rows** using the locked tree/epoch count. Test labels never enter refitting or early stopping. XGBoost early stopping now measures the same price MAE as candidate selection. Scalers are fitted separately on the tuning training rows and the final refit rows. LSTM evaluation windows retain past feature context even when an intermediate target is missing, preserving the same evaluation dates as Naive/XGBoost. The 60-step input represents retained feature observations; the data report explicitly counts gaps caused by incomplete rows. The inverse output transform and input scaler are embedded in each ONNX graph.

The first completed v1 study did not pass the declared model gates. [The experiment log](docs/MODEL_EXPERIMENTS.md) records its scores, the confirmed stale-fit/metric issues, and the v2 hypotheses. These previously reviewed folds are development evidence; repeated evaluation does not turn them into an independent test. The acceptance thresholds are unchanged. Reports now include per-fold/audit sample counts, consistent zero-return direction handling, prediction spread, train/validation drift and a paired 20-observation block-bootstrap interval for MAE gain versus Naive. An interval crossing zero does not establish a reliable gain.

Because the annual CFTC archive does not prove each report's actual release timestamp, CFTC values remain in the quality report but are **excluded from trained model inputs**. The new artifact disables the CFTC sensitivity control. No CFTC improvement is claimed. The feature-ablation report compares Cotton, Cotton+macro, and Cotton+macro+historical regime/volume/correlation features. Twenty-session block bootstrap MAE intervals, balanced accuracy, majority-direction reference, fold sample counts, hyperparameters, and LSTM loss/validation-loss curves are included in the artifact and Model Lab.

Each completed experiment is fingerprinted against its training source, protocol and train/validation data, then checkpointed directly in Drive. An interrupted LSTM trial without completed history restarts; a best-so-far checkpoint alone does not mark it complete. Normal Run All reuses its last successful Drive data cache, avoiding unnecessary Yahoo requests. An intentional refresh runs `python -m cottonlens_ml.prepare --drive-root /content/drive/MyDrive/CottonLensAI --refresh` inside the isolated Colab environment; refreshed source data receives a new fingerprint. ONNX exports are rejected when TensorFlow/ONNX parity reaches or exceeds `1e-4` maximum absolute error. LSTM SHAP values are precomputed in Colab; their approximation residual is reported instead of rescaling contributions to force an exact match. Neither TensorFlow nor SHAP is needed locally.

The exporter uses [native Keras ONNX export](https://keras.io/api/models/model_saving_apis/export/) rather than `tf2onnx.from_keras`. An inference-only CPU clone uses standard LSTM ops instead of cuDNN-only ops; the refit weights, 60-step two-output architecture and model selection are preserved. Both the wrapper and ONNX outputs are compared against the refit model with the refit-fitted scaler. There is no silent converter fallback: an incompatible stack stops at the smoke stage. XGBoost export contains the locked tree count fitted on the available pre-evaluation history so native backend and sklearn predictions agree.

The final cell validates the ZIP with the backend verifier and TensorFlow-free runtime, writes a checksum-bound receipt, then publishes `latest.txt` and downloads a text report. Keep the text and machine-readable JSON report, ZIP and SHA-256, validation receipt, source bundle, snapshot manifests, frozen cohort/readiness/selection, checkpoints and tracking evidence in Drive. The report includes data coverage, package/GPU versions, all four folds and paired metrics, candidate comparison, fixed validation trials and the descriptive audit. Feature ablation training is disabled on Day 1; its natural coverage is reported separately. A result that misses predefined gates is reported without changing them.

The resulting bundle is named similar to:

```text
cottonlens-model-v20260921-1430.zip
```

## Install a real Colab artifact

1. Copy the ZIP and its adjacent `.zip.sha256` file from Drive into `runtime/inbox/`.
2. Verify and install it with the backend container:

```bash
docker compose run --rm backend python scripts/import_artifact.py \
  /app/runtime/inbox/cottonlens-model-vYYYYMMDD-HHMM.zip \
  --destination /app/runtime/artifacts/current
```

3. Disable fixture seeding and rebuild/restart:

```bash
# PowerShell
$env:DEMO_MODE="false"
docker compose up --build
```

The installer verifies the outer ZIP digest before opening it, then verifies every packaged file before atomically replacing the active artifact. At startup, FastAPI verifies the artifact again, imports its Parquet history idempotently, removes development-fixture records, and loads only the selected XGBoost JSON or ONNX model. `GET /api/v1/health/ready` returns `503` when a validated artifact is missing or inconsistent and fixture mode is disabled.

## Data and evidence boundaries

- `CT=F`, `DX-Y.NYB`, and `CL=F` are downloaded through yfinance.
- `CT=F` is a convenient continuous-futures research proxy, not official ICE settlement data.
- CFTC Cotton No. 2 uses market code `033661` from annual Disaggregated Futures Only files.
- CFTC archive positions are not a model input until actual per-report publication timestamps can be verified; Friday-by-formula dates are not treated as proof.
- Historical reconstructed results are labelled `backtest`; they are not represented as forecasts that were recorded live.
- Sensitivity results hold other features fixed. They are model response, not causal inference.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/v1/health/live` | Process liveness |
| GET | `/api/v1/health/ready` | Database and artifact readiness |
| GET | `/api/v1/market/history` | Historical market observations |
| GET | `/api/v1/forecasts/latest` | T+1 and T+5 live forecasts |
| GET | `/api/v1/forecasts/history` | Live or backtest history |
| GET | `/api/v1/forecasts/{id}/explanation` | Local feature contributions |
| POST | `/api/v1/simulations` | Bounded sensitivity inference |
| GET | `/api/v1/models/metrics` | Model metrics and optional fold/learning-curve evidence |
| GET | `/api/v1/models/evaluation` | Walk-forward folds, feature ablation and selection audit; explicitly flags older artifacts |
| GET | `/api/v1/replay/{date}` | Point-in-time backtest audit |

Simulation bounds are enforced server-side: DXY ±5%, WTI ±20%, CFTC net position ±50,000 contracts, volatility multiplier 0.5–2.0.

## Development checks

Backend:

```bash
cd backend
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

Frontend requires Node 24.15+ for Angular 22:

```bash
cd frontend
npm ci
npm run build
```

On an older locally installed Node, the same build can be run with a temporary compatible runtime:

```bash
npx --yes node@24.15.0 node_modules/@angular/cli/bin/ng.js build
```

Model training is deliberately not part of local tests or CI. Tests use the labelled fixture and validate API behavior, additive explanations, simulation immutability, bounds, replay semantics, and artifact checksums.

## Five-minute demo

Turkish talk track and technical Q&A: [docs/INTERVIEW_DEMO.md](docs/INTERVIEW_DEMO.md).

1. Show T+1/T+5 forecast and the backtest chart.
2. Open **Why?** and explain `ŷ = E[f(X)] + Σφᵢ`.
3. Run a DXY/WTI scenario and point out the non-causal disclaimer; the new release disables CFTC sensitivity until publication dates are verified.
4. Compare Naive, Ridge, XGBoost and LSTM in Model Lab, including folds and LSTM training curves when the new artifact is installed.
5. Replay a historical forecast and distinguish it from a live record.
6. Close with the Colab → verified artifact → lightweight local runtime architecture.

## Known limitations

- No authentication, cloud deployment, weather/WASDE features, WebSocket, or LLM brief in this release.
- The project does not claim production trading suitability.
- LSTM sensitivity inference is available locally through ONNX, but new LSTM SHAP explanations are intentionally generated only in Colab.
- Docker verification requires Docker Desktop to be running. The application still has independent backend and frontend build/test paths.
