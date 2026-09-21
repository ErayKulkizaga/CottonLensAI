# CottonLens AI

Explainable Cotton No. 2 forecasting and market-sensitivity dashboard. CottonLens separates expensive model development from the lightweight interview demo:

- Google Colab performs ingestion, feature generation, XGBoost/LSTM training, MLflow tracking, holdout evaluation, explanations, and artifact export.
- The local stack only serves Angular, FastAPI, PostgreSQL, and CPU inference from a verified artifact.

The repository includes an explicitly labelled development fixture so the complete product flow can be reviewed before a real Colab artifact exists. Fixture values are never presented as trained results.

## Architecture

```text
Google Colab + GPU                    Local Docker
┌─────────────────────────┐          ┌───────────────────────┐
│ Yahoo/CFTC ingestion    │          │ Angular 22 + Nginx    │
│ leakage-safe features   │  ZIP     │          ↓            │
│ XGBoost + LSTM          ├─────────▶│ FastAPI inference     │
│ MLflow + holdout tests  │ verified │     ↙          ↘      │
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
3. Run all cells. The public repository URL is already configured in the notebook.

The notebook mounts Drive and writes to:

```text
MyDrive/CottonLensAI/
├── data/raw
├── data/processed
├── mlruns
├── checkpoints
└── artifacts/releases
```

The pipeline has a bounded search budget: at most 10 XGBoost configurations and 6 LSTM configurations. It uses a chronological 65/15/20 split and applies the documented model-selection rule only after candidates are locked.

Each completed experiment is fingerprinted against its train/validation data and checkpointed directly in Drive. Reconnecting to the same data resumes those candidates; refreshed source data receives a new fingerprint and is trained with the locked configuration. LSTM exports embed the train-fitted scaler in the ONNX graph and are rejected when TensorFlow/ONNX parity reaches or exceeds `1e-4` maximum absolute error. LSTM SHAP values are precomputed in Colab, so neither TensorFlow nor SHAP is needed locally.

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
- Tuesday CFTC positions become available to features on Friday; no backward filling is permitted.
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
| GET | `/api/v1/models/metrics` | Locked holdout metrics |
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

1. Show T+1/T+5 forecast and the backtest chart.
2. Open **Why?** and explain `ŷ = E[f(X)] + Σφᵢ`.
3. Run a DXY/WTI/CFTC scenario and point out the non-causal disclaimer.
4. Compare Naive, XGBoost and LSTM in Model Lab.
5. Replay a historical forecast and distinguish it from a live record.
6. Close with the Colab → verified artifact → lightweight local runtime architecture.

## Known limitations

- No authentication, cloud deployment, weather/WASDE features, WebSocket, or LLM brief in this release.
- The project does not claim production trading suitability.
- LSTM sensitivity inference is available locally through ONNX, but new LSTM SHAP explanations are intentionally generated only in Colab.
- Docker verification requires Docker Desktop to be running. The application still has independent backend and frontend build/test paths.
