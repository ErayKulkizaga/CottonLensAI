# CottonLens AI model card

## Purpose

CottonLens forecasts one- and five-session Cotton No. 2 log returns for a research and interview demonstration. It is not a trading system, an official ICE price source, or financial advice.

## Data

- Cotton No. 2 continuous-futures proxy: Yahoo Finance `CT=F`
- Dollar Index proxy: Yahoo Finance `DX-Y.NYB`
- WTI proxy: Yahoo Finance `CL=F`
- CFTC Disaggregated Futures Only, Cotton No. 2 market code `033661`

CFTC rows are retained for data auditing but excluded from model inputs until actual per-report publication timestamps are verified. DXY/WTI are delayed one Cotton session and forward-filled only from the past. The latest feature rows remain available for live inference even when their future targets are not yet known.

## Evaluation and selection

The next Colab release selects candidates on four 126-session pre-18-June-2024 rolling-origin folds. Five-session target purges protect each inner-validation and fold boundary. Protocol v2 selects settings with the most recent 126 valid inner-validation origins, then refits with all labels available before the evaluation period. Scalers are fit separately for tuning and refit, never with evaluation data. XGBoost early stopping uses price MAE. Budgets are eight XGBoost and four LSTM configurations per horizon/fold with fixed seed and early stopping; Ridge is a fixed-parameter reference. These folds have already informed revisions and count as development evidence.

For each horizon, a learned model must improve aggregate walk-forward MAE by at least 5% over Naive, reach directional accuracy of 53% for T+1 or 55% for T+5, and beat Naive in at least three of four folds. LSTM is selected over XGBoost only when aggregate MAE is at least 5% lower and directional accuracy is no worse, even if XGBoost fails its own gate. The previously seen 2024 onward period is descriptive only: it cannot select, reject or tune a candidate. If no learned model passes, the primary forecast is Naive and XGBoost sensitivity is labelled experimental.

Every new Colab release contains a generated model card with exact selected models, artifact version, fold evidence, training curves, and historical-audit metrics. The first completed v1 study selected Naive for both horizons; v2 market performance has not yet been measured. The currently installed older artifact predates this protocol and must be labelled legacy until replaced.

## Explainability

XGBoost uses native `pred_contribs` TreeSHAP. LSTM explanations use SHAP GradientExplainer and are precomputed in Colab. The actual model baseline is preserved; omitted feature mass is displayed as “Other features”. GradientExplainer's approximation residual is shown explicitly rather than being scaled away.

## Limitations

Continuous-futures roll construction, revised upstream data, regime shifts, missing weather/WASDE inputs, and proxy-source outages can materially affect results. Sensitivity simulation changes selected inputs while holding everything else fixed; it is not causal inference.
