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

## Result (16 Nov 2026, single run) — PRIMARY PASS
| Set | Hit rate (hits / pairs) | Season-block p | Binomial p |
|---|---|---|---|
| **B blind 1985–2011, US + TX (primary)** | **66.7% (62 / 93)** | **0.012** | 0.0009 |
| B, US only | 60.0% (36 / 60) | 0.106 | 0.078 |
| B, Texas only | 78.8% (26 / 33) | 0.003 | 0.0007 |
| B, all states pooled | 72.2% (298 / 413) | < 0.0001 | 4e-20 |
| B by month (US+TX): Sep→Oct / Oct→Nov / Nov→Dec | 69% / 71% / 61% | 0.057 / 0.029 / 0.178 | |
| R cotton 2012–2025, US + TX | 43.1% (28 / 65) | 0.87 | |
| C corn 2000–2025: US / states | 68.8% (33 / 48) / 57.0% (490 / 859) | 0.006 / 0.001 | |
| C soybeans 2000–2025: US / states | 66.7% (32 / 48) / 57.4% (455 / 793) | 0.029 / 0.001 | |

Excluded pairs with an unchanged forecast (zero revision): B 559, R 294, corn 693, soybeans 599. Unchanged state forecasts are common, especially for small states.

**Reading:**
- On never-seen cotton seasons 1985–2011, the direction of USDA's next revision was called correctly two times in three (Texas about four in five), with no fitted parameter.
- The same persistence holds in corn and soybeans 2000–2025, US and states. This is the smoothing documented by Isengildina-Massa et al. (2013) and the information rigidity of Goyal & Adjemian (2023), now shown as an out-of-sample directional call.
- In cotton since 2012 it does not hold: 43%, below chance. This matches the shrinking momentum coefficient and the regime change in Texas (T48).
- The finding is therefore historical and cross-crop, not a current cotton trading signal. The 2026 forward lock below is the live check.
- Evidence: `research/evidence/t50-revision-direction-20261116.json` / `.log`.
