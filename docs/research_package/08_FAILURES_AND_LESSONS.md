# 08 · Failures and lessons

## What failed, and why

| What | Phase | Why it failed | Lesson |
|---|---|---|---|
| Daily price models (XGBoost, CatBoost, Ridge, ElasticNet, LSTM, Quantile) | 1 | T+1/T+5 cotton returns behave close to a random walk for public features. Inner-validation gains did not transfer to outer years. | Measure against Naive with a hard margin. Report raw and shrunk results. |
| LSTM v1 | 1 | On the final validation split the best epoch was 1, then validation loss rose: too little signal and data for a recurrent model. | Sequence models need signal before capacity. |
| GPU placement in research-v1 | 1 | Keras built a CPU dataset inside a forced GPU scope. | Scope GPU placement to the model only; assert variable placement. |
| Early stopping on RMSE while selecting on MAE | 1 | Objective mismatch. | Align the early-stopping metric with the selection metric. |
| AMS annual futures workbooks | 1 | Wrong contract months, from the source itself, not the parser. | Verify contract identity against independent daily reports. |
| State models on weather/NDVI | 2 | Too coarse; levels overfit. | Use county anomalies against each county's own history. |
| USDA error corrections T14–T24 | 2 | Leak (final planted area) and unscaled ridge features. | Review every feature's publication time. Standardise inside the fold. |
| FSA-based claims (T29) | 2 | Some FSA files were saved after the USDA report. | File timestamps are evidence and need checking. |
| Hybrid of USDA yield × our area (T31, T32) | 2 | USDA's yield and area are estimated together; mixing ours with theirs breaks that consistency. | Do not splice components from different estimators. |
| GPU histogram CNN (T35) | 2 | About 2,700 labelled county-years; seed spread of 4 TX points. | Data size decides the architecture. Next step would be transfer learning from corn/soy. |
| Bias-correction ensemble (T36) on sealed 2022–2024 (T39) | 2 | USDA's US bias flipped (under-forecast in 2022 and 2024). | A bias term is a bet that the error regime persists. |
| State-specific bias (T40) | 2 | 1–12 years per state; the first folds went unshrunk. | Hierarchical shrinkage needs enough groups and years. |
| Texas FSA gate (T41) and Landsat failed-field share (T43) | 2 | In 2023, fields USDA later wrote off were still green at the 1 Sep and 1 Oct cutoffs. | Abandonment is often a later insurance/economic decision, not visible greenness. |
| Late-season (T37, T38) | 2 | USDA's December forecast (TX 4.7 / US 3.1) uses ginning and harvest data we only partly see before release. | Pick battles where the benchmark is weakest. |
| 2025 holdout of the bias correction (T42) | 2 | USDA under-forecast 2025, the third bias reversal. | Confirms the regime lesson. |
| RMA crop insurance as a feature | 2 | No publication date, so it leaks. | Reject a source before modelling if its timing can't be proven. |

## What worked
- **Revision momentum:** a sign-agnostic rule that follows USDA's own last revision. It beat USDA's average error at every report over 2002–2025, and held on blind 2002–2011 at Sep, Oct and Nov.
- **Process:** pre-declaration, sealed and blind tests, resumable cloud jobs, receipts and replay checks. These let the project reject its own false wins (T29, T36) before they reached anyone.
- **Risk lead (Phase 1):** the own-level HAR variance candidate passed its historical gates. Its forward test published twice and was stopped, so it is not forward-proven.

## Lessons for forecasting against an official benchmark
1. **Model the benchmark, not just the crop.** USDA's revision behaviour carried more usable signal than NDVI, drought or abandonment proxies.
2. **Separate "beats in development" from "beats out of sample".** Two of the project's development wins failed their sealed tests.
3. **Timing is the most common source of false skill.** File save times, revised archives and missing publication dates all produced wins that disappeared once checked.
4. **Report the failures.** They define where the edge is not, and they made the surviving result credible.
