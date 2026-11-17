# 14 · Who decided what, and what it cost

## Roles
- **Eray Kulkizaga** (project owner) set the goals and constraints and approved every irreversible step: opening sealed years, pushing to GitHub, using cloud jobs.
- **Codex** (Phase 1, 21 Sep – 7 Nov 2026) and **Claude** (Phase 2, 13–16 Nov 2026) wrote the code, ran the experiments and drafted the documents under those instructions.
- Every claim in this package was checked against files, and errors found later were corrected in the record (for example `12_CODEX_ANSWERS.md` and the dated amendments in each preregistration).

## Decision log (owner's decisions, from the project conversations)
| Date | Decision | Effect |
|---|---|---|
| Sep 2026 | Asked Codex for a hard target. Codex set +5% MAE gain vs Naive and direction gates of 53% / 55%. | Phase 1 pass rules (`2eb44a0`). |
| Oct–Nov 2026 | Approved the daily Windows forward job for the risk candidate (30 Oct), and later let it stop. | Two forward publications; no forward score. |
| 13 Nov 2026 | Switched the target to beating USDA's cotton production forecast. | Phase 2. |
| 15 Nov 2026 | "Ensemble + USDA fix", then late-season and ginning trials. | T36–T38. |
| 15 Nov 2026 | Approved opening the sealed 2022–2024 holdout after the pushed preregistration. | T39 (FAIL, recorded). |
| 16 Nov 2026 | Approved opening 2025. | T42 (FAIL). |
| 16 Nov 2026 | "Try everything, but watch for leakage; maybe combine untried methods in an ensemble." | Landsat T43, RMA screening (rejected for leakage), momentum T44/T45. |
| 16 Nov 2026 | Lock the 2026 forecast. | Forward lock, commit `18c3bc7`. |
| 16 Nov 2026 | Test the rule on corn and soybeans. | T46 (FAIL). |
| 16 Nov 2026 | Blind-test the rule on 1985–1999 cotton. | T47. |
| 16 Nov 2026 | Commissioned an external critical review, and asked that its gaps be fixed with evidence rather than with wording. | Robustness analysis, literature positioning, this file. |

Design choices made by the AI tools and accepted by the owner include:
- no-intercept momentum (after the bias-flip failure);
- the 2% / 3% parser gates;
- year-block permutation tests;
- the choice of preregistered pass rules.

## Compute and cost (what is recorded)
| Item | Hardware | Amount | Source |
|---|---|---|---|
| Phase 1 ledger fits | Colab T4 GPU (round A, market-v1), L4 in some November runs, local CPU | 32,934 fits; GPU hours not fully recorded | `research/trials.json`, `12_CODEX_ANSWERS.md` Q21 |
| `research-market-v1` ablation | Colab T4 | 44,085 s (12.25 h) fit compute, 20,360 sub-fits | ablation review JSON |
| Phase 2 histogram CNN | Kaggle GPU kernels | 120 trainings (T35 90, T39 30) | Kaggle outputs |
| Phase 2 satellite processing | Kaggle CPU kernels, Planetary Computer | 28,977 Landsat scenes (43,485 scanned), 2,278 MODIS composites | evidence JSONs |
| Phase 2 rolling refits | local CPU | about 1,000+ small refits (approx.) | `02_NUMBERS_CHEATSHEET.md` |
| T47 OCR | Kaggle CPU kernel | 50 scanned reports | `cottonlens-t47-ocr` |

- **Money:** Kaggle and Planetary Computer were used on free tiers. Colab spending is not recorded in the project.
- **Not recorded:** total GPU hours across Colab sessions.
- **Why the fit count is large:** each "fit" is one small model refitted at one origin or fold under a rolling design. A count of fits measures work done, not the number of independent hypotheses. The number of distinct experiments is 180: 133 in Phase 1 and 47 numbered trials in Phase 2 (T01–T47).
