# ADR 0001 · USDA cotton forecast monitor

Status: accepted, 16 Nov 2026. Code: `research/monitor/`. Log: `research/monitor/log.jsonl`.

## Context
The research produced frozen rules (momentum production correction, Texas area anchoring, next-revision direction) and two dated 2026 locks. To be useful and auditable, those rules must run on each new USDA report without human edits. Every output needs to be traceable to its exact inputs, model and code, and must not be changeable after the fact.

## Decisions
1. **Inputs from public ESMIS text reports, not the QuickStats API.**
   - No secret is needed, so the job can run in GitHub Actions or anywhere else.
   - Report text is what USDA actually published at release time; QuickStats values can be revised later.
   - Rejected: QuickStats. It needs a key, and its history is revised.
2. **Validated parsers with format-independent checks.**
   - Every table occurrence is tried, and the one with the most rows passing production = area × yield (±3%) wins.
   - Rejected: positional parsing. A September report has an area-only table with the same rows, and positional parsing read it as production (found while building this, on the Sep 2025 report).
3. **Gates with fallback to USDA's own number.** These gates decide per output whether the model runs or USDA's figure passes through, with the reason logged:
   - the states sum to the US within 2%;
   - Texas is present;
   - the previous report exists (otherwise the latest earlier one is used, as in research);
   - condition data is at most 21 days old;
   - revisions are within ±50%.

   Rejected: failing the whole run. A partial fallback keeps the safe outputs.
4. **Frozen model file** (`model_v1.json`).
   - It is trained through the 2025 season only and stored with its SHA-256 in every record.
   - A new model requires a new version file and is never edited in place.
5. **Intervals from realised out-of-sample errors.** The Texas area 80% / 95% intervals use the empirical quantiles of the anchoring model's rolling out-of-sample errors (T51, 2008–2025). Rejected: in-sample conformal residuals. T51 showed they under-cover (80% → 64%).
6. **Append-only, hash-chained JSONL log.**
   - Each record holds the SHA-256 of the previous one.
   - `verify` detects any later edit (unit test).
   - The git history adds a timestamp.
   - Rejected: a database. It would be heavier, and its edits are not visible in the repository.
7. **Idempotent runs.** A report already in the log is skipped, so the schedule can fire several times around a release day.
8. **Small first.**
   - Rejected for now: a REST API, a dashboard, Prefect/Dagster, MLflow and cloud deployment. The volume is five reports a year, and the log plus the static portfolio page serve the current users.
   - The existing FastAPI backend (`backend/`) can read the log later if an API is needed.

## Consequences
- **First live record:** the 1 Nov 2026 report, `status ok`. It reproduces the research locks exactly from the public text: US 12,763,510 bales, Texas 4,070,552 bales, Texas area 3,258,029 acres (80% interval 2.92–3.64 M).
- **Replay of Nov 2025:** this was the report after a cancelled October report and a pause in Crop Progress. The monitor used the September report for the revision and fell back to USDA for the Texas area because the condition data was stale. This is the intended behaviour.
- **Scheduling:** GitHub runs scheduled workflows only from the default branch. Until `usda-monitor.yml` is on `main`, runs are manual (workflow_dispatch) or local.
- **Known limits:**
  - The 50% revision guard can trip on very small states (New Mexico in Nov 2025). Their revision is then treated as zero.
  - Intervals come from 70 out-of-sample errors in one regime period.
