# CottonLens AI

Explainable Cotton No. 2 forecasting and market-sensitivity dashboard. CottonLens separates expensive model development from the lightweight interview demo:

- Google Colab performs ingestion, feature generation, Ridge/XGBoost/LSTM training, MLflow tracking, rolling-origin evaluation, explanations, and artifact export.
- The local stack only serves Angular, FastAPI, PostgreSQL, and CPU inference from a verified artifact.

The repository includes an explicitly labelled development fixture so the complete product flow can be reviewed before a real Colab artifact exists. Fixture values are never presented as trained results.

The artifact currently installed at `runtime/artifacts/current` was created with the **previous** evaluation protocol. The new walk-forward pipeline and Model Lab evidence become measured results only after a fresh Colab Run All and validated artifact import. This repository change alone does not improve an already-trained model or establish a new accuracy score.

## Architecture

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

1. Push or otherwise make this repository available to Colab.
2. Open `ml/notebooks/cottonlens_colab.ipynb` in a GPU runtime.
3. Select **Runtime → Run all**, approve Drive access, and leave the session running. The public repository URL is configured; make sure GitHub contains the notebook's matching code revision.

There is **one notebook with seven code cells**:

```text
Drive mount → clean repo clone/pull → isolated Python + locked dependencies
→ mandatory GPU/synthetic smoke → cached source validation → walk-forward training
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

The **next Colab run** uses four 126-Cotton-session rolling-origin folds ending before 18 June 2024. Every fold has an earlier training/inner-validation period, a five-session purge at both boundaries, and train-only transformations. Naive persistence and a fixed Ridge reference appear beside the learned models. The selected learned model must improve aggregate price MAE by at least 5% versus Naive, reach directional accuracy of 53% (T+1) or 55% (T+5), and beat Naive in at least 3/4 periods. LSTM displaces XGBoost only with another 5% MAE improvement and no directional-accuracy loss. The 2024 onward interval is a **previously observed historical audit**, not an untouched independent test: it may reject a locked candidate to Naive, but never trigger a search for another winner. No future accuracy is promised.

The Colab experiment budget is eight XGBoost hyperparameter configurations per horizon, four LSTM unit/dropout configurations, and fixed-configuration feature ablations. XGBoost has at most 1,200 trees with 50-round early stopping; LSTM has at most 100 epochs with 10-epoch early stopping. The 60-step LSTM carries prior feature context into each evaluation block, so its forecasts are measured on the exact same 126 dates as Naive/XGBoost; previous blocks' target labels are never used for fitting. Both input and output scalers are fit on training rows only. The inverse output transform and input scaler are embedded in each ONNX graph.

Because the annual CFTC archive does not prove each report's actual release timestamp, CFTC values remain in the quality report but are **excluded from trained model inputs**. The new artifact disables the CFTC sensitivity control. No CFTC improvement is claimed. The feature-ablation report compares Cotton, Cotton+macro, and Cotton+macro+historical regime/volume/correlation features. Twenty-session block bootstrap MAE intervals, balanced accuracy, majority-direction reference, fold sample counts, hyperparameters, and LSTM loss/validation-loss curves are included in the artifact and Model Lab.

Each completed experiment is fingerprinted against its train/validation data and checkpointed directly in Drive. Normal Run All reuses its last successful Drive cache, avoiding unnecessary Yahoo requests. An intentional refresh runs `python -m cottonlens_ml.prepare --drive-root /content/drive/MyDrive/CottonLensAI --refresh` inside the isolated Colab environment; refreshed source data receives a new fingerprint. ONNX exports are rejected when TensorFlow/ONNX parity reaches or exceeds `1e-4` maximum absolute error. LSTM SHAP values are precomputed in Colab; their approximation residual is reported instead of rescaling contributions to force an exact match. Neither TensorFlow nor SHAP is needed locally.

The exporter uses [native Keras ONNX export](https://keras.io/api/models/model_saving_apis/export/) rather than `tf2onnx.from_keras`. An inference-only CPU clone uses standard LSTM ops instead of cuDNN-only ops; original fitted weights, 60-step training sequences, multi-output training and model selection are unchanged. Both the wrapper and ONNX outputs are compared against the original model with the train-fitted scaler. There is no silent converter fallback: an incompatible stack stops at the smoke stage. XGBoost export keeps only the early-stopping-selected trees so the native backend and sklearn predictions agree; its training/search/selection policy is unchanged.

The final cell verifies the ZIP with the existing backend importer, checks the exported live forecasts against the TensorFlow-free backend runtime, and prints/downloads `cottonlens-results-vYYYYMMDD-HHMM.txt`. The same text file and complete machine-readable `.json` evidence are retained under `MyDrive/CottonLensAI/reports/`. The report includes data quality, exact package/GPU versions, all four folds and aggregate metrics, confidence intervals, Naive/Ridge/XGBoost/LSTM comparison, feature ablations, all validation trial settings/results, the selected LSTM epoch curve, historical-audit outcomes, selection-gate reasons, export formats and ZIP digest. Send **only this text report** in chat for model review; keep the ZIP and `.sha256` in Drive until the final release is approved for local integration. A result that misses the predefined gates is reported honestly rather than hidden or endlessly re-tuned against the already viewed audit period. All shell commands are orchestrated by the notebook; no local training command is required.

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
