# CottonLens AI model card

## Purpose

CottonLens forecasts one- and five-session Cotton No. 2 log returns for a research and interview demonstration. It is not a trading system, an official ICE price source, or financial advice.

## Data

- Cotton No. 2 continuous-futures proxy: Yahoo Finance `CT=F`
- Dollar Index proxy: Yahoo Finance `DX-Y.NYB`
- WTI proxy: Yahoo Finance `CL=F`
- CFTC Disaggregated Futures Only, Cotton No. 2 market code `033661`

CFTC Tuesday positions become usable on Friday. Other sources are aligned to Cotton trading dates with forward fill only. The latest feature rows remain available for live inference even when their future targets are not yet known.

## Evaluation and selection

Data is split chronologically into 65% train, 15% validation, and 20% locked test. Scalers are fit only on train. Candidate budgets are capped at ten XGBoost and six LSTM configurations with fixed seed and early stopping.

For each horizon, a learned model must improve MAE by at least 5% over Naive on both validation and the one-time locked test. Directional accuracy must be at least 53% for T+1 and 55% for T+5 on both splits. LSTM is selected over a qualifying XGBoost model only when its locked-test MAE is at least 5% lower and directional accuracy is no worse. If neither learned model passes, the primary forecast is Naive and the XGBoost sensitivity model is labelled experimental.

Every Colab release contains a generated model card with the exact selected models, artifact version, split ranges, and locked holdout metrics.

## Explainability

XGBoost uses native `pred_contribs` TreeSHAP. LSTM explanations use SHAP GradientExplainer and are precomputed in Colab. Only the eight largest displayed contributions are returned; omitted contribution mass is folded into the displayed base value so the UI equation remains additive.

## Limitations

Continuous-futures roll construction, revised upstream data, regime shifts, missing weather/WASDE inputs, and proxy-source outages can materially affect results. Sensitivity simulation changes selected inputs while holding everything else fixed; it is not causal inference.
