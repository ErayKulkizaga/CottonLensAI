# Forward lock: 2026 upland cotton production (locked 16 Nov 2026)

- **Rule.** The revision-momentum rule (T44 spec, unchanged): e_hat = b · log(USDA_m / USDA_prev) / sd, forecast = USDA_m · exp(e_hat).
- **Coefficients.** Frozen from seasons 2000–2025 only:
  - Nov: b = 0.01011, sd = 0.08305;
  - Dec: b = −0.00567, sd = 0.07315.
- **Scoring.** Against NASS's 2026 final (expected May 2027), compared with USDA's November and December 2026 forecasts.

## Locked now (from USDA November 2026, released 10 Nov 2026)
| | USDA Nov 2026 | Locked forecast |
|---|---|---|
| United States | 12,794,000 bales | **12,763,510 bales** |
| Texas | 4,100,000 bales | **4,070,552 bales** |

State-level values: `research/evidence/forward-lock-2026-20261116.json`.

## December procedure (applied mechanically when USDA's December report is released, 10 Dec 2026)
Per state: forecast = USDA_Dec × exp(−0.00567 · log(USDA_Dec / USDA_Nov) / 0.07315). US = USDA_US_Dec × Σ state forecasts / Σ state USDA_Dec.

## Notes written at lock time
- **The corrections are small by design.** The rule moves USDA's number by a fraction of its latest revision; here that is −0.2% for the US.
- **The December coefficient is slightly negative on the full 2000–2025 data,** although the rolling record favoured momentum at Dec (14/7 years better for the US). The 2026 December correction will therefore be very small and opposite to the Nov→Dec revision. This is reported as is; the rule is not changed.
- **Track record of this rule (2002–2025, rolling, 24 seasons):** lower average error than USDA at every monthly report for TX, the US and production-weighted states. It does not win every year. See `docs/CROP_NOWCAST_TRIALS.md`.
