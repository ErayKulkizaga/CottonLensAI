# CottonLensAI research plan (16 Nov 2026)

This plan merges three external reviews into one. It supersedes `docs/archive/phase2-notes/SENIOR_ROADMAP_20261116.md`.

## 1. What the reviews ask for
| Review | Core demand |
|---|---|
| 1 · "How hard is it really?" | Literature positioning; honest statistics (multiple comparisons, dependence, extreme years); static result tables; clear ownership of decisions; compute that is documented, not just counted. |
| 2 · "Path to senior" | Few, hypothesis-driven, preregistered tests instead of many models. Decompose the error mechanism. Compare against simple alternatives. A point-in-time production system with a fallback and a log. Forward results reported unchanged. |
| 3 · "Path to exceptional" | New, independently verifiable knowledge about **why** USDA errs. Ideally a mechanism that public data can see **before** the official report, validated in independent seasons and other settings. The target is a mechanism, not a better number. |

**Common thread:** stop asking "can we beat USDA's number?" Ask instead: **"which production risk does USDA see late, why, and can public data see it earlier?"**

## 2. Where the project stands (evidence already committed)
- **Engineering and protocol** (strong): publication-time rules; sealed and blind holdouts; pushed preregistrations with dated amendments; invalid runs kept; resumable cloud jobs; parser validation (153/153 and 441/441 exact).
- **Momentum rule:**
  - 2002–2025 mean error below USDA in 12/12 cells.
  - Not significant per cell after Holm/BH correction; pooled p = 0.015 (a test defined after the per-cell results were seen).
  - The coefficient shrinks over time.
  - It does not beat a published-style adjustment in 6 of 8 cells.
  - Failed on corn and soybeans (T46) and on blind 1995–1999 (T47 stage A).
  - **This is a replication of published work, not the contribution.**
- **The lead for a real contribution: error mechanism** (T48, `15_USDA_ERROR_MECHANISM.md`).
  - In Texas the dominant error source shifted around 2012, from yield to **harvested area**: the mean area error rose from 3.8% to 14.9% at 1 Sep.
  - USDA's in-season area barely moves (the Dec error is 79–98% of the Aug error).
  - USDA has mostly over-estimated harvested area since 2012, i.e. under-estimated abandonment (2023: −52%).

## 3. The central research question
> **H-core ("anchored abandonment"):** in-season, USDA's Texas harvested-area forecast stays anchored to planted area × a typical harvest ratio, and responds too little to drought information that is public at the cutoff. The final harvested area does respond to it. So the in-season area error is predictable from public drought and crop-condition data **available before each report**.

**Why this can be new:**
- Prior work (Isengildina-Massa et al. 2012, 2013; Goyal & Adjemian 2023) studies the correlation of revisions in production or yield.
- We found none that decomposes cotton's error into area and yield, documents the post-2012 shift, or links the area error to observable abandonment drivers.
- This must be checked again with a targeted literature search before any claim (step M0).

**What would falsify it:**
- USDA's in-season harvest ratio already tracks drought as strongly as the final ratio does; or
- the area error is not predictable from information available at the cutoff once the season mean is removed; or
- the relation does not hold in independent settings.

## 4. Workstreams

### A. Auditability (days 1–2)
- **A1.** All portfolio tables rendered as static HTML; the 2026 locked number written into the text.
- **A2.** `research/reproduce.py`: rebuilds every headline number from committed evidence and cached public data, diffs it against the published values and exits non-zero on mismatch. Pinned environment file.
- **A3.** One data-version index: every source with its SHA-256, retrieval time and publication-time rule.
- **A4.** Finish T47 stage B (OCR 1985–1994) and report it whatever it shows.

### B. Mechanism study (days 2–4)
- **M0. Literature check.** A targeted search for harvested-area or abandonment error in USDA cotton forecasts, and for FSA and administrative-data changes around 2012. Record what exists.
- **M1. Descriptive, Texas, 2000–2025.**
  - Compare USDA's in-season harvest ratio (harvested ÷ planted at each report) with the final ratio.
  - Measure the sensitivity of each ratio to public drought at the cutoff: the share of Texas cotton area in USDM D2+; NASS cotton condition (% poor/very poor); June-to-cutoff rainfall.
  - Prediction of H-core: the final ratio's sensitivity is much larger than USDA's in-season sensitivity.
- **M2. Alternative explanations, each tested:**
  - (a) a change in the final-area method (FSA administrative acreage), seen in final-revision patterns and NASS methodology notes;
  - (b) a shift of planting to dryland (dryland share);
  - (c) more frequent extreme droughts. Does the error per unit of drought change, or only the frequency of drought?
- **M3. Preregistered prediction test.**
  - Model: an area-error model with at most two public predictors, fixed form, rolling fit on past seasons only, information strictly before each report.
  - Compared with USDA, the momentum rule and the published-style adjustment, on production as well as area.
  - Metrics: mean error, a year-block test, and leave-one-season-out, with extreme years reported separately.
  - Uncertainty: an interval for the area error (rolling split-conformal), with coverage and width reported.
- **M4. Independent generalisation test, preregistered and frozen form.** The same mechanism where abandonment matters but which we have never examined:
  - **winter wheat** harvested area in Kansas, Oklahoma and Texas (USDA May–Aug forecasts, large drought abandonment);
  - dryland cotton states other than Texas (Oklahoma, Kansas).
  - A pass rule is written before any data is pulled.

### C. Production system (days 4–6)
- **S1. USDA Forecast Monitor:**
  - Detects a new ESMIS Crop Production release.
  - Parses it with the validated parser and gates.
  - Applies the locked rule (and the M3 area model if it passes) and attaches the interval.
  - Falls back to USDA when a gate fails or inputs drift.
  - Writes an append-only, hash-chained log.
- **S2.** Scheduled run (GitHub Actions; ESMIS text needs no secret), and a measured latency from release to logged forecast.
- **S3. First live run:** the 10 Dec 2026 report, under the frozen procedure in `docs/FORWARD_LOCK_2026.md`. The result goes into the log unchanged.

### D. Communication (day 6–7)
- **D1. Portfolio page and package rewritten around the mechanism:**
  - what USDA sees late (Texas harvested area since 2012);
  - what public data can or cannot see earlier;
  - where corrections help and where not to use them.
- **D2. Interview answers** updated, including the limits.
- **D3. The owner rehearses.** One reviewer asked whether the person behind the project understands it, and only rehearsal can answer that. A self-test sheet: 20 questions with short model answers, each pointing to its evidence file.

### E. After the interview (months to years)
- **E1.** 2026 forward score in May 2027; 2027 season run live through the monitor.
- **E2.** If M3/M4 pass: a working paper (AgEcon Search / SSRN) with code and data versions, and a note to the authors of the prior work.
- **E3.** Generalise toward the reviewers' "theory" question: when do official production forecasts under-react to public information? Use the same decomposition on other USDA crops and on other official forecasts.

## 5. Rules for every step
1. A pass rule is written and pushed before the data that tests it is pulled.
2. One run per test. Amendments are dated and limited to data or parsing fixes, and invalid runs are kept.
3. Every number shown has a static table and an evidence file.
4. Claims stay within the evidence. A failure is reported as prominently as a success.
5. Few tests, each motivated by a stated information gap.

## 6. Honest limits of this plan
- **"Exceptional" cannot be promised.** M3/M4 can fail. If they do, the result is still a rigorous finding: public data cannot anticipate USDA's Texas area error earlier, for the reasons measured in M1/M2.
- **Independent seasons are finite.** Cotton seasons since 1995 have now all been seen. New cotton evidence comes only forward (2026, 2027, …) or from other crops and regions (M4).
- **Mechanism ≠ cause.** M1/M2 can rule explanations in or out. Causal claims need the alternative-explanation tests and stay hedged.
