# Roadmap after the external review (16 Nov 2026)

> Superseded by `docs/RESEARCH_PLAN_20261116.md` (merged plan after three external reviews).

Two external reviews judged the project as substantial engineering with modest scientific novelty.
- **Their main points:**
  - the smoothing idea is published;
  - the edge does not survive multiple-comparison correction;
  - large fit counts are not evidence;
  - results rendered by JavaScript looked empty to text readers;
  - who decided what was unclear;
  - there is no production system.
- **Their advice:** do not train more models. Diagnose *why* USDA is wrong, test a few preregistered hypotheses, measure uncertainty, and build an auditable forecasting system.

This file records the plan. Each item ends in a committed artifact. A claim is made only if its evidence exists.

## Already done (16 Nov)
- **Literature and contribution:** `docs/research_package/13_LITERATURE_AND_CONTRIBUTION.md`.
- **Robustness of the momentum record:** `research/crop_nowcast/robustness.py`, `research/evidence/robustness-momentum-20261116.json`. Covers Holm/BH correction, a year bootstrap, leave-one-out, the coefficient path, and a published-style adjustment compared on the same seasons.
- **Who decided what, and compute:** `docs/research_package/14_DECISIONS_AND_COMPUTE.md`.
- **Preregistered generalisation tests:** T46 corn/soy (FAIL); T47 blind 1985–1999 (stage A FAIL; stage B OCR running).

## Priority 1 — auditable
- **Static tables:** every result table on the portfolio page is rendered in static HTML. JavaScript only adds interactivity.
- **One command to reproduce:** `research/reproduce.py` rebuilds every headline number from the committed evidence and cached public data, and checks them against the published values. A pinned environment file goes with it.
- **Data versions:** each source file has a SHA-256 and a retrieval time (existing receipts), indexed in one table.

## Priority 2 — the error mechanism
- **Decomposition.** log(final / USDA) = log(harvested_final / harvested_USDA) + log(yield_final / yield_USDA), by state, month and season.
  - Data: QuickStats area and yield forecasts for 2012–2025; area and yield columns from the parsed reports for 1995–2011.
  - Questions:
    - Which part dominates in Texas, and in the US?
    - When does USDA learn about abandonment?
    - Does the momentum effect come from the area part or the yield part?
- **Deliverable:** `docs/research_package/15_USDA_ERROR_MECHANISM.md` with static tables and charts.

## Priority 3 — few, strong, preregistered tests
- **H1, "when to trust USDA" (uncertainty, not point accuracy).** Information available at each report predicts the size of USDA's remaining error:
  - the size of the latest revision;
  - the month;
  - the Texas share of production;
  - the area part of past revisions.
  - **Test:** calibrated intervals (rolling split-conformal), checked on blind seasons for coverage and width against a constant-width baseline.
- **H2 (only if Priority 2 supports it).** A signal targeted at Texas abandonment, tested on seasons not used to build it.
- Each test gets a pushed preregistration, one run, and a full report.

## Priority 4 — an automated, point-in-time system
- **USDA Forecast Monitor:**
  - detect a new Crop Production release on ESMIS;
  - parse it with the validated parser and its gates;
  - apply the locked rule;
  - attach the interval;
  - fall back to USDA's number when a gate fails or inputs drift;
  - write an append-only, hash-chained log.
- **Runs on a schedule** (GitHub Actions, no secrets needed for ESMIS text).
- **First live run:** the 10 Dec 2026 report, under the frozen December procedure in `docs/FORWARD_LOCK_2026.md`.
- **Measured:** latency from release to logged forecast, failure handling, and cost.

## Priority 5 — forward results
- **2026 lock, unchanged:** scored in May 2027 against the NASS final, reported whatever it shows.
- **Stated limit:** one season cannot prove an edge. The forward log is how evidence accumulates.

## Framing going forward
The project is presented as a diagnosis of where and why USDA's cotton forecasts miss, and of when a correction helps. It is backed by preregistered tests, including the failures, and by a system that produces auditable forecasts. It is not presented as "beating USDA".
