# Corrected V2 · Day 1 validation record

Date: 2026-09-24. This document describes implementation and local no-fit verification. It is **not** a measured corrected-V2 market result or a successful Colab runtime run.

## Evidence boundaries

The V1 text report survives, but the original raw/processed snapshots, exact 504 evaluation origins, tracking/checkpoints and full release ZIP have not been recovered locally. The installed runtime artifact is a separate legacy release. Corrected V2 freezes its own source SHA-256 inventory (including the uncommitted working tree), raw/processed Parquet snapshots, exact 504 development origins, protocol and environment identity before model fitting. The previously seen 2024+ audit is descriptive only and cannot select, veto or tune a model. The 126-origin refit policy remains locked; 21 is an unvalidated hypothesis for a future development-period comparison.

## Local Day 1 checks

Run `python -m pytest ml/tests -q` with `ml/src` on `PYTHONPATH`; this suite uses synthetic data and fake estimators only. Run `cd backend && python -m pytest -q && python -m ruff check .`; backend tests set a unique temporary SQLite URL at collection time and guard against workspace DB access. Run `python -m ruff check ml/src ml/tests ml/source_bundle.py ml/colab_setup.py ml/inference_probe.py`. With Node 24.15+, run `cd frontend && npm run build` using installed lockfile dependencies. Local actual training, full walk-forward scoring and Colab compatibility fits are forbidden.

The Colab notebook consumes `output/cottonlens-day1-source.zip` plus its checksum, runs the mandatory Python 3.12/GPU/synthetic parity smoke, validates cached sources, then runs `cottonlens_ml.preflight`. A frozen `ready.json` is required before the Colab-only pipeline can start. Set `RUN_TRAINING=True` only after reviewing preflight. Export remains pending until `validate_release` confirms checksum and backend runtime parity; only then is `latest.txt` updated and a report generated. Preserve the source bundle, immutable snapshots, cohort, readiness, pre-audit selection, tracking/checkpoints, ZIP/checksum, validation receipt and TXT/JSON report.

## Limits

No trusted ICE exchange calendar is integrated. Historical horizons count the next one/five recorded Cotton observations; a provider omission cannot be distinguished from a true exchange holiday. Live future target dates remain unknown (`null`) until an actual observation exists. Daily external source bars use a conservative next-UTC-day availability assumption with one Cotton-observation lag and a three-observation age bound; exact vendor publication times are not independently proven. The Colab GPU, pinned Python 3.12 environment, TensorFlow/ONNX compatibility and real data coverage remain to be observed by the notebook smoke and preflight before any market result is accepted.
