# CottonLens AI contributor notes

- GPU/sequence training and `cottonlens-train` remain Colab-only; CI never trains real models. The approved full-year pilot may run small CPU tabular/HAR/GARCH fits locally with explicit `COTTONLENS_ALLOW_LOCAL_CPU_TABULAR=1`, one process and at most two threads. Use the same ledger and a separate CPU dependency group; no TensorFlow/CUDA installation locally.
- Statistical Naive/drift/ARIMA pilot fits remain Colab CPU-only, one process/at most two threads. They are not added to the local CPU allowlist. CI may test synthetic data; it cannot fit real market data. No automatic ARIMA order search or silent numerical fallback.
- Keep 5% price-MAE and 53%/55% direction gates fixed: legacy four-block protocols require 3/4 wins; full-year eight-block protocols require 6/8. Already-reviewed history is research evidence; 2024+ audit cannot guide selection. Current canonical decisions: `docs/MASTER_PLAN_20261001.md`.
- Tune with purged recent validation, then refit only on labels available before each evaluation origin. Keep checkpoint identities tied to data and recipe; incomplete trial files are not completed experiments.
- Keep TensorFlow, MLflow, Jupyter, SHAP, CUDA, and training-only packages out of `backend/pyproject.toml` and the backend Docker image.
- Treat `data_quality=illustrative` as a development fixture; never describe it as measured performance.
- Preserve the `live` versus `backtest` distinction in API and UI changes.
- Any artifact schema change must update the Colab exporter, checksum verifier, importer, runtime loader, and contract tests together.
- Backend checks: `cd backend && python -m pytest -q && python -m ruff check .`.
- Frontend check with Node 24.15+: `cd frontend && npm ci && npm run build`.
- Before completing UI changes, render at 1440, 1024, and 390 px and confirm `scrollWidth - clientWidth === 0` at each width.
