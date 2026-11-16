# T50 preregistration: can the direction of USDA's next cotton revision be called in advance?

Written 16 Nov 2026, before any next-revision sign was computed. The monthly forecast series used here were loaded before, in T45, T46, T47 and the momentum record, to compute errors against finals. The month-to-month persistence of revision signs was never tabulated. The rule below has no fitted parameter.

## Rule (no parameters)
At report m, predict the sign of USDA's next revision, sign(log(F_{m+1} / F_m)), to equal the sign of its latest revision, sign(log(F_m / F_prev)).
- prev = the latest earlier monthly forecast, as in `momentum.with_revision`.
- Pairs scored: Sep→Oct, Oct→Nov, Nov→Dec. For example, at the 1 Oct report, the Sep→Oct revision predicts the sign of the Oct→Nov revision.
- Unchanged forecasts are excluded: if either revision is exactly 0, the pair is not scored. The number excluded is reported.

## Data, cotton upland production
| Set | Seasons | Source | Role |
|---|---|---|---|
| **B (blind eras)** | 1985–1999 and 2000–2011 | reports parsed in T47 (gated) and T45 | **primary** |
| R (recent) | 2012–2025 | QuickStats | secondary |
| C (other crops) | corn and soybeans 2000–2025 | T46 data | secondary, generalisation |

Units: the **US** total and **Texas** (primary), and all states pooled (secondary).

## Metric and pass rule
- **Hit rate** = share of scored pairs where the predicted sign equals the actual sign. Chance is 50%.
- **Test:** one-sided season-block permutation. Within each season, all predictions are flipped together (20,000 draws, seed 42). The statistic is the hit rate pooled over US and TX pairs.
- **PRIMARY PASS:** on set B, the pooled US + TX hit rate is above 50% with permutation p < 0.05.
- **Also reported:** a binomial p (pairs treated as independent, optimistic), each unit and month separately, sets R and C, and states pooled.

## Forward lock
Before the 10 Dec 2026 Crop Production report, the predicted sign of the Nov→Dec revision is locked for the US and Texas, together with the Nov→Dec revision directions of every state that has a Nov 2026 forecast (= sign of the Oct→Nov 2026 revision). They are scored on 10 Dec 2026. Evidence: `research/evidence/t50-*.json`.
