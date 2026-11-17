# 07 · Evaluation protocol and metrics

## Metrics
- **MAE gain vs Naive (price):** 100 × (MAE_naive − MAE_model) / MAE_naive. Positive means better. Target ≥ +5%.
- **Direction accuracy:** the share of origins where the sign of the predicted return matches the realised sign.
  - Gates: 53% (T+1), 55% (T+5).
  - "All-origin direction" counts flat (zero) forecasts as misses. "Active direction" uses only non-zero forecasts.
- **Active rate:** the share of origins where the shrunk forecast differs from Naive.
- **QLIKE (risk):** L(σ̂², r²) = r²/σ̂² − log(r²/σ̂²) − 1. Robust to noise in the variance proxy. Gains are reported as baseline − candidate.
- **MAPE (production):** mean |forecast / final − 1| × 100 over years, separately for TX and the US.
- **Production-weighted state error:** Σ |error_s| · final_s / Σ final_s over 17 states. This catches US-level cancellation across states.
- **Years better / worse:** a count of seasons where |error_model| < |error_USDA|.

## Splits and refits
- **Rolling origin (walk-forward):** for each test year Y, fit only on data available before Y (Phase 2), or on mature labels before each origin (Phase 1). There is no random K-fold anywhere.
- **Phase 1 nested validation:** 8 outer annual blocks (2016–2023); 3 inner blocks of 63 origins before each; early-stopping segments separate from scored inner blocks.
- **Label maturity:** 5 sessions before a T+5 label can be used. Refit every 21 observations (some designs 126).
- **Purges** between inner and outer blocks for overlapping T+5 labels.
- **Shrinkage selection:** final = Naive + w·(model − Naive), w ∈ {0, .25, .5, .75, 1}, chosen on inner blocks only. Raw (w = 1) results are always reported too.
- **Phase 2:** county models refit per test year on years < Y. USDA corrections are pooled over states with fold-wise standardisation. The ensemble weights for Y use only the members' out-of-sample predictions from years < Y.

## Uncertainty
- **Moving-block bootstrap:** blocks of 20 and 60 observations, 10,000 draws, seed 42, fold boundaries respected. A pass needs both CIs above 0.
- **Year counts** (for example 6/8 positive years), plus a top-1% trimmed sensitivity for risk.
- **Seed spread** for the neural networks (5 seeds, min–max).
- **Phase 2 caution:** with 3–12 years per test, results are reported with year-by-year detail. A 3-of-3 win is about a 1-in-8 event under a coin-flip null.

## Pre-declaration and holdouts
- Every trial's pass rule was written and committed before the run. Phase 2 sealed tests were pushed to GitHub before unsealing.
- Amendments made before outcomes were seen are dated (for example the FSA earliest-snapshot rule fix, and the T45 decimal parser fix).
- Results are committed as produced, failures included. Claims later found leaky were withdrawn in writing (T14–T24, T29).
- About 60 Phase 2 variants were tried in development; the best single cells were treated as optimistic, never as claims.

## Leakage controls that caught real problems
1. **Final planted area used in-season** (T14–T24): replaced with June Acreage.
2. **FSA September/October files saved after USDA's report** in some years: T29 claim withdrawn.
3. **Crop insurance data with no publication date:** source rejected.
4. **AMS annual workbooks with wrong contract months:** named-market results marked compromised.
5. **Decimal parsing in old reports** (would have read yield as production): caught by a unit check before scoring.
6. **Yahoo price revisions:** the first-observed cohort design.
