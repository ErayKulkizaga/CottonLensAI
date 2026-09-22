# Colab stabilization — verification record

Date: 2026-09-22. This is an implementation/check record, **not a successful Colab training report**.

## Results

| Check | Evidence |
|---|---|
| Lightweight ML regression tests | 23 passed on local Python 3.13; no training imports/fit calls |
| Existing backend tests | 9 passed |
| ML lint | Passed using the repository backend Ruff configuration |
| Python compilation / tracked diff whitespace | Passed |
| Python 3.12 Linux locked dependency resolution | `uv sync --locked --extra cuda --python 3.12 --python-platform x86_64-manylinux_2_35 --dry-run` passed; no installation/training performed |
| Single notebook orchestration | Seven code cells parse; stage order and isolated setup commands tested |
| Backend lint | Three existing I001 import-order failures in `migrations/env.py`, `tests/test_api.py`, `tests/test_artifacts.py`; unchanged |
| Colab GPU detection, actual LSTM fit/save/load/native ONNX export and parity | **Not run: no connected Colab runtime available** |
| Actual XGBoost fit/save/load, MLflow nested experiment/Drive restore, isolated inference probe | **Not run: mandatory Colab smoke implements these checks** |
| Full Run All, real source ingestion, final artifact validation | **Not run: must execute in Colab** |

Exact direct versions are in `constraints/colab-py312.txt` and README; the entire resolved graph is in `uv.lock`. They remain a candidate runtime stack until the real smoke passes. Native Keras conversion is not represented as proven; failure prevents full training. No local TensorFlow installation, WSL startup, model training, Docker rebuild, backend/frontend edits or GitHub push was performed for this change.

## Changed and added files

Paths below are repository-relative; braces group files with the same purpose.

| Files | Purpose |
|---|---|
| `README.md`, `ml/VALIDATION.md` | Run All instructions, exact pins, recovery policy, evidence and limitations |
| `ml/notebooks/cottonlens_colab.ipynb` | The only notebook; seven sequential guarded code cells |
| `ml/colab_setup.py` | Targeted uv bootstrap, managed Python 3.12.11, isolated locked training/inference environments, streaming errors |
| `ml/pyproject.toml`, `ml/constraints/colab-py312.txt`, `ml/uv.lock` | Python boundary, exact direct and transitive dependency versions |
| `ml/src/cottonlens_ml/__init__.py` | Agg and TensorFlow Keras backend before imports |
| `ml/src/cottonlens_ml/tracking.py` | Local SQLite with consistent Drive snapshots, single-writer guard and restore |
| `ml/src/cottonlens_ml/{pipeline,training}.py` | Tracking integration; existing training, selection and checkpoint logic retained |
| `ml/src/cottonlens_ml/{quality,features,prepare}.py` | Explicit invalid-price report, deterministic past-only processing, source validation before training |
| `ml/src/cottonlens_ml/{export,onnx_export,xgb_export}.py` | Embedded scaler and raw-input ONNX parity gates; selected-tree XGBoost serialization |
| `ml/src/cottonlens_ml/{smoke,validate_release}.py`, `ml/inference_probe.py` | Synthetic Colab checks and exact backend importer/runtime compatibility checks without TensorFlow in inference |
| `ml/tests/{test_features,test_colab_contract,test_tracking_snapshot,test_xgb_export}.py` | Regression tests; no local training |

## Next verification

Make this exact code revision available to the notebook's configured GitHub repository, then open the single notebook with a T4 runtime and Run All. Drive authorization is interactive; no terminal commands are needed. Inspect `MyDrive/CottonLensAI/reports/environment-smoke.json`: it must say `passed` and contain measured T+1/T+5 parity errors under `1e-4`. Only use the ZIP printed by the successful last cell. If conversion fails, retain the full traceback and adjust/test the converter stack in Colab before claiming compatibility; do not bypass smoke or relax parity.
