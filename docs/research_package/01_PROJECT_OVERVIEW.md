# 01 · Project overview

## One paragraph
CottonLensAI is a research and application project on US cotton markets. Phase 1 (with Codex) asked whether public data can forecast ICE cotton futures prices one and five trading days ahead better than a no-change forecast. Over 133 experiments and about 32,900 logged model fits the answer was no: the best selected result was +0.22% MAE gain against a +5% target. The phase did produce a confirmed risk (volatility) research line, a live forward-testing system, and an event study showing cotton futures move 1.42× more on USDA report days. That pointed Phase 2 (with Claude) at the USDA Crop Production report itself.

Phase 2 asked whether US and Texas upland cotton production can be forecast better than USDA's monthly forecast. After 45 pre-declared trials (satellite gradient boosting, Landsat field analysis, a GPU CNN, USDA error corrections, ensembles, two sealed holdouts), one fixed rule held up: USDA revises cautiously, and a no-intercept "revision momentum" correction had lower mean error than USDA at every monthly report from 2002 to 2025 (12 of 12 report × region cells), including 10 seasons tested blind. The 2026 crop forecast is locked in Git for a forward test in May 2027.

## Timeline

| Date (2026) | Phase | Milestone |
|---|---|---|
| 21 Sep | 1 | First commit: application (FastAPI backend, Angular frontend, ML package) |
| 23–25 Sep | 1 | v1 XGBoost vs LSTM price models (Colab); corrected v2; tabular sprint. All select Naive. |
| 30 Sep – 1 Oct | 1 | Research engine (ledger, receipts, nested validation). `research-v1` stopped by a Keras GPU-placement bug; `research-v2-tf-placement` round A: 7,331 XGBoost fits on a T4 GPU. `research-market-v1` ablation: 20,360 sub-fits, 12.2 compute hours. |
| 2 – 9 Oct | 1 | Source explorations (WASDE, NASS, FAS, CFTC, FX, weather, crop prices, AMS, on-call): 1,855 fits each. Availability clock, return path, agri transfer, regional WASDE/NASS T+1 pilots. Weak-signal power control: 3,380 synthetic fits. |
| 10 – 12 Oct | 1 | Contract-curve T+1/T+5, paired MSE vs MAE loss, MAE calibration, selection stability, named-contract (same-contract label) controls, CFTC source control. |
| 12 Oct – 4 Nov | 1 | Risk/volatility track: HAR, EWMA, GARCH under QLIKE; weak-teacher control (1,044 synthetic fits); signed shocks; DXY risk state; own-level candidate (7/8 positive years); live daily forward publishing; pre-2010 same-contract replication; AMS contract-identity defects found. |
| 5 Nov | 1 | Price-only conditional median (Quantile) T+1: 338 + 4 fits, STOP. GitHub consolidation. |
| 6 – 7 Nov | 1 | Data admission and GPU (L4) infrastructure audits. |
| 12 Nov | — | Handoff to Claude review workspace. |
| 13 – 14 Nov | 2 | Crop nowcast data contract; MODIS NDVI county features; state and county models (T01–T13); Landsat field features (Move 3). |
| 15 Nov | 2 | USDA error corrections (T14–T27, leak found and fixed), FSA, hybrids, late season, GPU histogram CNN on Kaggle (T35), ensemble (T36), Nov/Dec and ginning (T37–T38). Link 1 event study. |
| 15 – 16 Nov | 2 | Sealed 2022–2024 test (T39): fail. State-specific bias (T40), Texas FSA gate (T41), 2025 holdout (T42), Landsat failed-field test (T43): all fail. |
| 16 Nov | 2 | Revision momentum (T44), blind 2000–2011 test (T45), full 2002–2025 record, 2026 forward lock, research page and this package. |

## Headline results

### Phase 1: price and risk
- **Price (T+1, T+5):** no model beat the naive no-change forecast by the pre-set +5% MAE margin. Best registry result +0.22% (Ridge, fundamental joint, T+5). Median of 113 scored experiments −0.65%. Best ablation cell +0.76% (XGBoost, expanded availability, T+5, direction 57.5%, 5/8 folds).
- **Synthetic controls** (4,597 fits): the pipeline recovers an injected signal and does not invent one under the null, so the price failures are not a pipeline artefact.
- **Risk:** an own-level HAR candidate improved QLIKE over EWMA by +0.0177 with both block-bootstrap CIs positive and 7/8 positive years (historical candidate, not forward-proven). A live forward cohort of 252 origins runs to 2 Nov 2027.
- **Event study:** cotton futures daily moves are 1.42× larger on USDA Crop Production report days (p = 0.004, 38 events). USDA's own revision did not predict the sign of the move (Spearman −0.23, p = 0.08).

### Phase 2: production vs USDA

| Report | Rule TX MAPE | USDA TX | Rule US | USDA US | Rule states | USDA states |
|---|---|---|---|---|---|---|
| 1 Sep | 11.9 | 12.9 | 6.0 | 6.3 | 10.5 | 11.0 |
| 1 Oct | 11.3 | 12.1 | 4.9 | 5.5 | 9.1 | 9.3 |
| 1 Nov | 7.5 | 8.0 | 3.5 | 3.6 | 6.2 | 6.4 |
| 1 Dec | 5.0 | 5.3 | 2.3 | 2.5 | 4.3 | 4.5 |

Revision-momentum rule, 2002–2025 rolling, 24 seasons (22 at Oct). "States" is the production-weighted error over 17 states.
- Seasons closer than USDA (better / worse): Sep TX 10/9, US 9/10; Oct TX 9/9, US 12/9; Nov TX 13/8, US 12/10; Dec TX 9/8, US 14/7.
- The formal pre-declared tests did not pass. T44 required Sep as well; T45 tied 3–3 in TX at Dec. The full record is descriptive evidence; 2002–2011 were blind.
- The earlier bias correction (T25/T36) passed development, then failed both sealed tests because USDA's bias reversed in 2022, 2024 and 2025.
- 2026 forecast locked 16 Nov 2026 (commit `18c3bc7`): US 12,763,510 bales vs USDA November 12,794,000.

## Who did what
- **Eray Kulkizaga:** owner, research direction, all go/no-go decisions (unsealing, pushing, approving trials).
- **Codex** (Phase 1) and **Claude** (Phase 2): coding agents that implemented pipelines, ran experiments on Colab and Kaggle, kept the ledgers and wrote the documents, under the owner's rules: pre-declare, never change success criteria silently, keep failures, ask before costly or irreversible steps.
