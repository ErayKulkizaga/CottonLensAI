# CottonLens AI contributor notes

- Never train models locally or in CI. Run `cottonlens-train` only in Google Colab.
- Keep the 5% price-MAE, 53%/55% direction and 3-of-4 fold gates fixed when iterating; document results and treat already-reviewed folds/audit as development evidence.
- Tune with purged recent validation, then refit only on labels available before each evaluation origin. Keep checkpoint identities tied to data and recipe; incomplete trial files are not completed experiments.
- Keep TensorFlow, MLflow, Jupyter, SHAP, CUDA, and training-only packages out of `backend/pyproject.toml` and the backend Docker image.
- Treat `data_quality=illustrative` as a development fixture; never describe it as measured performance.
- Preserve the `live` versus `backtest` distinction in API and UI changes.
- Any artifact schema change must update the Colab exporter, checksum verifier, importer, runtime loader, and contract tests together.
- Backend checks: `cd backend && python -m pytest -q && python -m ruff check .`.
- Frontend check with Node 24.15+: `cd frontend && npm ci && npm run build`.
- Before completing UI changes, render at 1440, 1024, and 390 px and confirm `scrollWidth - clientWidth === 0` at each width.
