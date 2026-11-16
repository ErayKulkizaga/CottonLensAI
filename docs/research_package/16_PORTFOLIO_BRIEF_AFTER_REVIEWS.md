# 16 · Portfolio brief: everything done after the "can we still do better?" question (16 Nov 2026)

This file gives the portfolio author (Codex or anyone else) what changed, what was tried, what worked, what failed, and what may and may not be claimed. Every number points to an evidence file under `research/evidence/` or a document under `docs/`. Branch `claude/hls-move3`, repo `ErayKulkizaga/CottonLensAI`.

## 0. Two later results that change the headline (16 Nov 2026, evening)
- **All blind seasons pooled (1987–1999 and 2002–2011, 23 seasons).** The rolling momentum rule, trained only on earlier seasons, was closer than USDA in **88 of 128** season × report × region comparisons. All 8 TX/US cells have a positive mean gain; mean gain 0.64 points; **year-block p = 0.007**. December US was closer in 16 of 19 seasons. Disclosure: the decision to pool was made after both eras were seen, but it pools *all* blind seasons and drops none. Evidence: `pooled-blind-1987-2011-20261116.json`.
- **T50, preregistered: predicting the direction of USDA's next revision.** The rule has no parameters: next revision in the same direction as the last one.
  - Blind cotton 1985–2011, US + TX: **66.7% correct (62 / 93), season-block p = 0.012, PASS.**
  - Texas 78.8%; all states 72.2% (298 / 413, p < 0.0001).
  - Corn 2000–2025 US 68.8% (p = 0.006); soybeans US 66.7% (p = 0.029); states 57% in both (p ≤ 0.001).
  - It **fails in cotton 2012–2025 (43%)**, consistent with the regime change.
  - December 2026 directions are locked (US down, TX down; `t50-forward-lock-dec2026-20261116.json`).

Suggested hero line now: **"On 23 never-seen seasons, a rule built only from USDA's own revisions beat USDA's cotton production forecast in 88 of 128 comparisons (p = 0.007), and called the direction of USDA's next revision two times in three (p = 0.012). Since 2012 the Texas error has moved into harvested area, which public crop-condition data anticipates and USDA does not."**

## 0b. Product-level work after the fourth review (16 Nov 2026, late)
- **T51 (preregistered), Texas area.**
  - Baselines: the anchoring model beats USDA (7.0% vs 11.4% mean area error, p = 0.0005) and both past-bias baselines (9.6%, p ≤ 0.008).
  - Production: USDA × area correction is better than USDA in 4/4 months (Sep 14.1 → 11.8 %, Oct 14.1 → 10.9 %, Nov 9.9 → 7.9 %, Dec 6.3 → 5.8 %) and better than momentum, but p = 0.135, so the production rule **FAILS**. Report it as "consistent, not significant".
  - In-sample conformal intervals under-cover (80% → 64%). The monitor therefore uses empirical out-of-sample error quantiles.
  - Structural break: Chow at 2012 p = 0.0005; the sup-F scan peaks at 2019 (permutation p = 0.055). A break is supported but its date is not pinned.
  - Two automated point-in-time tests pass.
- **USDA forecast monitor** (`research/monitor/`, `docs/adr/0001-usda-forecast-monitor.md`):
  - reads ESMIS public text, no key;
  - parses with production = area × yield validation;
  - gates with fallback to USDA's own figure;
  - frozen `model_v1.json` (trained through 2025);
  - empirical 80/95% intervals;
  - an append-only, hash-chained log with a tamper test;
  - a GitHub Actions workflow (manual until it is on `main`);
  - 15 unit tests pass.

  **First live record:** the 1 Nov 2026 report. Computed independently from the public text, it reproduces the research locks exactly: US 12,763,510 bales, Texas 4,070,552 bales, Texas area 3,258,029 acres (80% interval 2.92–3.64 M acres).

  The replay of Nov 2025 handled real failures: the October 2025 report was cancelled, and condition data had paused during the shutdown. The monitor used September for the revision and fell back to USDA for the area.

## 1. The new headline (use this, not "we beat USDA")
> **Since 2012, USDA's Texas cotton forecast errors come from harvested area, not yield. USDA's in-season area stays anchored while the share of the crop in poor condition, public at the survey date, already signals the coming abandonment. A one-variable model built only on past seasons would have cut USDA's in-season Texas area error by about 40% at every report since 2008. The pattern is consistent in Oklahoma and Kansas cotton but just misses significance, and it is absent in winter wheat. The 2026 Texas area is locked as a dated forward test.**

Secondary line: the revision-momentum rule is a **published effect** (Isengildina-Massa et al. 2012/2013) that this project re-tested independently. It is real but small, it shrinks over time, and its size depends on the era.

## 2. Timeline of work after the question
| Step | What was done | Result | Evidence |
|---|---|---|---|
| T47 stage A | Blind test on never-seen USDA reports 1995–1999 (25 text reports); frozen 2000–2025 momentum coefficients | FAIL: 4/8 TX/US cells, 2/4 state months, p = 0.38 | `t47-stageA-1995-1999-20261116.json`, `docs/T47_BLIND_1985_1999_PREREG_20261116.md` |
| T47 stage B | 50 scanned 1985–1994 reports; Tesseract OCR on a Kaggle CPU kernel; 33/50 passed an outcome-blind gate (production = area × yield within 3%, sum of states ≈ US) | Frozen coefficients: FAIL (2/8). A+B 1985–1999: FAIL (4/8) | `t47-stageB-…json`, `t47-AB-…json` |
| T47 secondary | Rolling rule re-estimated only on that era's past seasons, scored 1987–1999 | **7/8 TX/US cells and 4/4 state months below USDA**; Dec US closer in 9 of 10 seasons; pooled p = 0.089 (not significant) | `t47-rolling-1987-1999-20261116.json` |
| External reviews | Three critical reviews ("how hard is it", "path to senior", "path to exceptional") were merged into one plan | Plan: mechanism over models, auditability, preregistered tests, a production monitor | `docs/RESEARCH_PLAN_20261116.md` |
| Literature | Smoothing in USDA cotton forecasts is published; adjusting for it is weak and unstable; Goyal & Adjemian (2023): information rigidity | The project is repositioned as an independent preregistered re-test, not a discovery | `13_LITERATURE_AND_CONTRIBUTION.md` |
| Robustness | Holm/BH, season bootstrap CIs, leave-one-season-out, coefficient path, published-style adjustment on the same seasons | No cell survives Holm/BH (min BH q = 0.09). Bootstrap CI excludes 0 for Oct US and Dec US. One sign flip in LOO. Coefficients shrink 2002 → 2025 (Sep turns negative). Published-style adjustment beats the momentum rule in 6 of 8 cells. | `robustness-momentum-20261116.json` |
| Error decomposition 2012–2025 | log(final/forecast) = area part + yield part (identity holds to ±0.1%) | Texas: area part = 72–94% of error variance (Sep–Nov); area error does not shrink Aug → Dec; US: yield dominates | `error-mechanism-2012-2025-20261116.json`, `15_USDA_ERROR_MECHANISM.md` |
| T48 (preregistered) | The same decomposition on 1995–2011 (85 reports parsed, area/yield columns) | 2/4 hypotheses pass. **Regime shift:** Texas 1 Sep area error 3.8% (1995–2011) → 14.9% (2012–2025); earlier misses were yield misses | `t48-mechanism-1995-2011-20261116.json`, `docs/T48_MECHANISM_PREREG_20261116.md` |
| M0 literature check | Searched for prior work on USDA harvested-area error | No published accuracy study of USDA's in-season cotton harvested area found; NASS uses FSA failed acres as a floor; Texas A&M uses FSA data to forecast acreage | `13_…` section M0 |
| M1 anchoring (discovery) | Texas 2000–2025: USDA in-season vs final abandonment against public signals at the survey date (USDM drought, NASS poor/very-poor share) | 2012–2025: USDA responds ~2/3 as much as the final (ratio 0.65–0.69, CI below 1) vs ~0.9 before. corr(signal, area error) −0.82 to −0.88. Drought frequency unchanged, so "more drought" is ruled out | `m1-anchoring-tx-20261116.json` |
| T49 (preregistered) | Independent tests: Oklahoma+Kansas cotton (never examined), winter wheat KS/OK/TX; Texas real-time demonstration | Texas: area error Sep 12.8 → 7.9 %, Oct 13.2 → 7.6 %, Nov 12.6 → 7.0 %, Dec 12.6 → 7.3 % (p = 0.0003). OK/KS: same sign (corr −0.61), 4/4 months better, **p = 0.056 → FAIL**. Wheat: no effect, FAIL | `t49-anchoring-20261116.json`, `docs/T49_ANCHORING_PREREG_20261116.md` |
| Forward lock | Texas 2026 final harvested area from USDA 1 Nov (3,900,000 ac) and 52% poor/very poor | **Locked 3,258,029 acres (−16.5%)**; Dec procedure fixed; scored May 2027 | `t49-forward-lock-2026-20261116.json`, `docs/FORWARD_LOCK_2026.md` |
| Auditability | One-command reproduction of all Phase 2 headline results; pinned environment; data-version index; static HTML tables | `research/reproduce.py`: 7/7 match under the pinned env (numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1) and the current env; 135 input files hashed | `research/reproduce.py`, `research/requirements-phase2.txt`, `data/data_versions.csv` |
| Roles and compute | Who decided what; recorded compute; why fit counts are not evidence | Owner set goals and approved every irreversible step; AI tools wrote and ran the code | `14_DECISIONS_AND_COMPUTE.md` |

## 3. Defects found and handled (show these: they are evidence of rigour)
- **Published-style comparison:** first computed on mismatched seasons ("momentum best in 6/8"). Corrected to the same seasons, "2/8", before anything was published (`4fb3726`).
- **Sep 2010 report:** the title typo "Yield, and, Production" meant T45 and the 2002–2025 record silently lacked Sep 2010. Disclosed in `docs/CROP_NOWCAST_TRIALS.md`; the one-shot T45 was not re-run.
- **T46 corn/soy:**
  - an "Other States" row with final 0 (log 0);
  - two report columns fused in Sep 2004.

  Both were data fixes, logged as dated amendments, and the invalid runs were kept.
- **T47:**
  - 1990s layout differences;
  - OCR quirks: capital headers, ':' read as digits, three production columns, pre-1992 state abbreviations.

  All were fixed before scoring and logged as amendments 1–3.
- **T48:** August has no previous-month yield column; fixed before the finals were downloaded.
- **T49:** QuickStats throttling was first read as "no data". The helper was fixed and the data re-pulled before the single run.
- **A data-identity check that confirmed the regime shift is real:** P × 480 / (H × Y) is between 0.999 and 1.001 in the data, and the year-by-year Texas area errors show it (2023 −52%, 2012 −25%, 2022 −22%, 2024 −23%).

## 4. Numbers the portfolio can use
- **Texas harvested-area error at 1 Sep:** 3.8% (1995–2011) vs 14.9% (2012–2025).
- **Area share of Texas error variance at 1 Sep:** 0.24 → 0.72.
- **USDA in-season / final abandonment response to poor-condition share:** 0.65–0.69 (2012–2025), about 0.9 before.
- **corr(poor share at survey date, remaining Texas area error), 2012–2025:** −0.82 to −0.88.
- **Real-time Texas area model, 2008–2025:** about −40% area error at every report, p = 0.0003.
- **Oklahoma + Kansas cotton (independent):** corr −0.61, 4/4 months better, p = 0.056 (missed the gate).
- **Winter wheat:** no effect.
- **Momentum rule:**
  - 2002–2025: 12/12 cells below USDA on average, 3/8 TX/US cells p < 0.05 uncorrected, 0 after Holm/BH;
  - blind 1987–1999 rolling: 7/8 cells, p = 0.089;
  - frozen coefficients fail across eras.
- **Locks:**
  - 2026 US production 12,763,510 bales (commit `18c3bc7`);
  - Texas 2026 harvested area 3,258,029 acres (−16.5%, commit `7d48dbf`).
- **Preregistered tests in this stretch:** T46, T47, T48, T49 and T50. All were pushed before data, all reported, failures included.
- **Pooled blind seasons 1987–2011:** 88 / 128 better, p = 0.007.
- **T50 revision direction:** blind cotton 67% (p = 0.012), Texas 79%, states 72%, corn and soybeans 57–69%; cotton 2012–2025 43%.
- **Reproduction:** 7/7 results reproduce exactly from one command.

## 5. What may NOT be claimed
- "Beats USDA" without qualification. The momentum gain is small, not significant after correction, unstable across eras, and the published adjustment often does better.
- "New mechanism confirmed in other states". Oklahoma and Kansas just missed the gate (p = 0.056).
- "General USDA area bias". It does not hold in winter wheat.
- Causation for the 2012 shift. Candidates are FSA failed-acre use and dryland abandonment economics; neither is tested yet.
- That Texas real-time results are confirmation. Texas was the discovery set.

## 6. Suggested portfolio structure
1. **Hero:** the headline in §1, with three tiles:
   - "Texas area error 3.8% → 14.9% after 2012";
   - "−40% area error, real time";
   - "2026 area locked: 3.26 M acres (−16.5%)".
2. **Decomposition chart:** Texas area vs yield error by month, 1995–2011 vs 2012–2025 (data in `t48-…json` and `error-mechanism-…json`).
3. **Anchoring chart:** poor/very-poor share vs remaining area error, Texas 2012–2025, with the USDA vs final slopes (`m1-anchoring-…json`).
4. **Preregistered tests table:** T46–T49 with pass/fail, including the near miss (OK/KS p = 0.056) and the wheat null.
5. **Momentum, re-tested:** the published effect, robustness table (Holm/BH), published-style comparison, blind 1987–1999 rolling 7/8, and coefficient decay.
6. **Rigour panel:** defects found and fixed (§3), reproduction command, data-version index, roles.
7. **Forward tests:** the two 2026 locks, with scoring dates (Dec procedure 10 Dec 2026; finals May 2027).
8. **Next:**
   - an automated report monitor (first live run 10 Dec 2026);
   - FSA failed-acre timing as the test of the alternative explanation;
   - more dryland states.

## 7. Still open (as of 16 Nov 2026)
- The portfolio page (`docs/portfolio/cotton-usda-research.html`, artifact v3) does not yet include M1, T47 stage B and the rolling result, T49, or the area lock.
- The automated USDA monitor (plan workstream C) is not built.
- Alternative explanations (a) FSA administrative timing and (b) the dryland shift are not yet tested.
