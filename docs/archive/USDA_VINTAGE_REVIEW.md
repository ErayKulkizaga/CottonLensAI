# USDA vintage review status

## Verified acquisition

The local acquisition check recorded 15 FAS marketing-year responses (2010–2024,
27,887 rows) and 14 NASS calendar-year responses (2010–2023, 5,974 rows).
All requested years contained records and their raw SHA256 checks passed. Duplicate
checks were within each annual response, not across marketing-year boundaries.
These are current API snapshots, not certified historical vintages.

## First numerical reconciliation

NASS report: <https://www.nass.usda.gov/Publications/Todays_Reports/reports/prog2320.pdf>.
Release page: <https://esmis.nal.usda.gov/publication/crop-progress/2020-06-01>.
The release page links a TXT version as well as the PDF. Preserve those actual links;
do not invent report URLs from an assumed weekly schedule.

For week ending 2020-05-31, seven current national Cotton observations match pages
4–5 exactly: planted 66%, squaring 8%, and condition categories 1/7/48/39/5%.
This validates only those seven values. It does not validate every API row, the
availability of five-year averages, or the history of revisions.

The archive adapter now retrieves the release page and the TXT report from the
page's actual link, hashes both, and extracts the five national Cotton condition
percentages. One week in each year 2010–2023 passed the strict five-value comparison
against Quick Stats. This is 14 checked weeks out of the much larger archive, not
proof that every week is complete or unchanged. The result is saved in
`output/reports/nass-cross-year-vintage-pilot.json`.

The repeatable full-year audit compared all 23 current national Upland condition
weeks in 2023 against each week's archived ESMIS TXT. All 115 category values
matched. Memorial Day and Labor Day weeks resolved to Tuesday release pages;
the audit records attempted dates and the actual linked report. Evidence and
the checksummed result are under `output/usda-vintage-2023-audit/`. The command
is `python -m cottonlens_ml.sources.nass_archive --audit-snapshot SOURCE_JSON
--output AUDIT_JSON` from the ML environment. This broader numerical match still
does not establish each report's exact public availability time or that all
historical revisions have been found.

The official Crop Progress description and report revision policy state release
after 4:00 p.m. Eastern Time on the first business day. The ESMIS release page
shows a date; its HTML `datetime` (for example 12:00Z) must not be read as the
actual market-availability clock. The adapter explicitly leaves the clock and
model eligibility unverified. Holiday/delay and correction evidence remains
necessary for the strict historical protocol.

The separate official NASS monthly publication calendar matched all 23 audited
2023 release dates to a `Crop Progress` row scheduled for `4:00 pm ET`, with
status `Published` at retrieval. The hashed calendar pages and cross-check are
under `output/usda-nass-calendar-2023/`. This verifies the listed schedule and
current status, **not** the actual first-publication instant or absence of later
corrections. The calendar auditor counts a row as matched only when the report
link, date, scheduled time and status all agree; it always leaves
`model_eligible=false`.

`cottonlens_ml.sources.vintage` validates evidence hashes, exact record selectors,
country aggregation units and report rounding before producing a diagnostic.
It always leaves `model_eligible=false`; the existing publication package review
and compiler remain the only route to historical feature admission.

The FAS indexed report for week ending 2020-05-28 contains rounded Cotton totals
consistent with API totals, but its direct official PDF URL returned HTTP 404 on
2026-09-28. Search-index text is not accepted as the immutable source evidence.

## AMS physical spot quotation discovery

The keyed MyMarketNews slug 3804 historical query returned no 2020/2023 records,
but this does **not** mean historical USDA spot quotes are unavailable. The
archived MyMarketNews report catalog lists 3804 as `Daily Spot Cotton Quotations`
with a 2026 report date; it has no `MP_CN001` entry. Its `published_date` field
cannot be assigned to the older ESMIS excerpt reports. The
separate official ESMIS [Daily Spot Quotations, excerpts](https://esmis.nal.usda.gov/publication/daily-spot-quotations-excerpts)
archive has linked `MP_CN001.TXT` reports. The new `cottonlens_ml.sources.ams_archive`
adapter checks each report date, the `41-4/34` and `31-3/35` quality labels, and
the published seven-market `AVERAGE` row before immutably archiving the page and
TXT with SHA256 receipts. Eight single-date samples from 2018–2025 parsed; for
example the `41-4/34` average was 43.23 cents/lb on 2020-04-01. The samples live
in `output/usda-ams-spot-pilot/`. They do not establish continuous coverage.

The monthly listing audit uses each official `View` link, including date-suffixed
variants, instead of constructing release URLs from the date. It preserves each
HTML listing page by hash and refuses ambiguous rows or repeated releases.
April 2020 has 21 listed releases; all 21 linked TXT reports passed date, quality
and average-row parsing. Three of those reports had zero *reported spot bales*
while still publishing quotations, confirming that quotations are not observed
transaction-weighted prices. The inventory and audited files are under
`output/usda-ams-month-audit/`. Re-running `--audit-inventory` reuses only
checksum-verified complete local reports and rejects a corrupted cached file.

A separate 12-month 2020 listing inventory found 247 unique report dates versus
262 weekdays (`output/usda-ams-2020-inventory/year-inventory-2020.json`).
The saved listing pages can now reconstruct monthly inventories offline with
`--month YYYY-MM --year-inventory YEAR_JSON --output INVENTORY_ROOT`; this checks
the saved HTML hashes and pagination before any report download. All 12 months
were subsequently audited: 247/247 linked TXT reports passed strict parsing and
their report SHA256 checks. Twelve reported zero spot bales despite publishing a
quotation. The aggregate receipt is
`output/usda-ams-2020-inventory/year-content-audit-2020.json`; the 800 evidence
files were copied to the private Drive reports directory and matched byte-for-byte
by SHA256. This is one year of content coverage, not full historical coverage or
proof of historical publication time.
The 15 weekday gaps include holidays; direct release-page probes
also returned HTTP 404 for 2020-02-13, 2020-02-14, 2020-11-20 and 2020-12-02.
These four cannot be imputed or assumed to be market closures. The official AMS
[2025 report redesign notice](https://www.ams.usda.gov/content/usda-revising-daily-spot-cotton-quotations-report)
also means the historical TXT layout cannot silently be treated as the post-August
2025 DSQ format.

The resumable `ml/prepare_ams_history.py` run subsequently completed every
listed `MP_CN001` report in 2021 (**252**), 2022 (**250**) and 2023 (**250**).
Together with 2020, `ml/export_ams_diagnostic.py` rechecks all archived TXT
files and emits **999 unique dated rows**, 2020-01-02 through 2023-12-29, in
`output/ams-diagnostic/spot-quotations-2020-2023.csv`. The immutable sidecar
records the table SHA256, annual source-audit hashes and exporter-code hash.
Zero reported spot bales occur in 12/25/48/8 reports by year, without removing
their published quotations. The 2021-10-20 excerpt has a literal masked
`#########` dateline; a separate official `MP_CN002` front page confirms its
date, both quotations and bale total. The 2023-04-28 TXT contains Windows-1252
punctuation; the raw bytes and strict decoding path are preserved. Other
unknown date/format changes still fail closed.

The source files, audit receipts, CSV, manifest, and relevant code/tests are
in `output/ams-diagnostic/ams-spot-evidence-2020-2023.zip` (3,242 members,
SHA256 `ce794337a041cfab83e8670d4528a7701a98e24aec6aa5f1d53aee0790d7e9df`).
The ZIP, receipt, CSV and manifest were byte-verified after copying to the
private synced Drive folder `reports/ams-diagnostic`. This is a **diagnostic
table**, not a reviewed as-published feature package; no model was trained.

The [oldest ESMIS archive listing](https://esmis.nal.usda.gov/publication/daily-spot-quotations-excerpts?page=170)
starts on 2018-11-21, so this particular source cannot cover the 2016–2017
research blocks. November 2018 yielded nine links for six dates with identical
same-date TXT hashes. In March 2019, one 2019-03-28 page links to a TXT dated
2019-03-27 while its `-0` page links to a correctly dated report. The conflicting
page and TXT were quarantined under `output/usda-ams-2019/quarantine`; 2019 and
2018 are **not** declared complete or added to the 999-row table. A future
comparison using this shorter source needs a separately frozen common cohort.

These are physical spot *quotations* for specified cotton quality, not Cotlook A,
ICE open interest, a transaction-weighted spot price, or the CT=F prediction target.
The ESMIS `12:00Z` HTML field is not an actual publication clock. All receipts
therefore retain `published_at=null` and `model_eligible=false` until release timing,
missing dates, corrections and source-use terms are reviewed.

### MMN publication/version pilot (2026-10-01)

The official [report archive](https://mymarketnews.ams.usda.gov/filerepo/reports?field_slug_id_value=3004&field_report_date_end_value=2023-07-31)
was accessible in the browser. Its 2023-07-31 row displays `07-31-2023 02:20:30 pm`,
status `Final`, and a versioned TXT link under
`/filerepo/sites/default/files/3004/2023-07-31/1029205/`.
The downloaded file SHA256 is
`549a5c8e5aacd68cac8d02fc194dfb4cc146d42f0c546e2efbc3c7ff22258f23`,
identical to the frozen ESMIS sample. This establishes one file-version match;
it does not certify all 999 records or an original-versus-corrected vintage.

The displayed table clock has no explicit timezone. For 2025-07-31, the archive
shows `01:45:31 pm` while the official public index gives epoch
`1753987531803` and `12:45:31 MDT`. This is an observed offset cross-check,
not authority to assume a timezone for every historical row. Preserve the
literal clock and keep `published_at=null` until the clock convention is reviewed.
Local direct HTTP archive requests still timed out; browser navigation/download
worked. The legacy data API is not a fallback for pre-August-2025 reports.

`cottonlens_ml.sources.ams_publications` now inventories all slug-3004 pages,
retains all listed versions/statuses, and downloads each relevant version to
compare its exact bytes with the frozen 999-row diagnostic table. Pagination,
date filters, report identity, paths, hashes and interrupted-download resume
are checked. Every missing or different version remains explicit. The completed
review is immutable and remains model-ineligible; file equivalence alone cannot
close timezone, first-version or usage review. No schedule-derived clocks are added.

Use the existing **Data Workbench**, `SOURCE_ACTION='ams_publication_review'`,
on a CPU runtime. It writes `reports/ams-publication-history-v1` in private Drive,
requires no API key, reuses completed verified downloads, and never launches
Research Workbench or modifies `research-market-v1`. It prints page and
50-file progress, not full origin/model records. Local pilot evidence lives in
`output/ams-publication-probe-20261001/`.

USDA's [2025 Cotton Price FAQ](https://mymarketnews.ams.usda.gov/sites/default/files/resources/2025-08/FAQs-Cotton-Price-Reporting-and-Data-Access_508.pdf)
explicitly says pre-August-2025 cotton price data are absent from the MyMarketNews
data API. It identifies `MP_CN001` as archived report slug **3004**, distinct from
post-redesign data slug 3804. The public *Search Previous Reports* interface exposes
`Published Date`, `Report Date` and report status; those could provide independent
per-report timing evidence if matched to the ESMIS payloads. The archive search
timed out in two bounded local probes on 2026-09-28, so no historical clock or
version has been certified from it. Do not infer one from the modern API catalog.

## Blocking evidence still needed

The official NASS Corrections table was archived on 2026-09-28 with SHA256
`0252c77998228986f29f169df7d694fd7f61c695996ce0c0d0991d159faefb1d`.
It contains eight Crop Progress notices, including the 2021-04-05 delay to
17:00 ET and the 2022-11-28 rescheduling to 2022-11-29. The parser preserves
source text and fails on changed table structure. Command:
`python -m cottonlens_ml.sources.nass_corrections --output output/nass-corrections`.
This inventory provides concrete exceptions to reconcile with archived reports;
it is not a complete revision history, and absence of a notice does not certify
an unchanged first-release vintage. Raw evidence is mirrored in private Drive
`reports/nass-corrections`. Model eligibility remains false.

Exception reconciliation now covers all eight listed notices. Eight corresponding
official TXT reports/pages were archived; header dates match their page dates,
including explicitly rescheduled dates. The pilot archive audit verified 24
release dates and matched 8/8 notices. The separate 2023 archive audit verified
23 dates with no matching notice in this snapshot; that absence is not evidence
of no revisions. Both report/page hashes and the correction-table hash are checked.
Reproduce with `nass_corrections --notice-folder <correction-snapshot>
--archive-root <archive> --output <reports>`; immutable reconciliation files live
in `output/nass-corrections/reconciliation` and private Drive reports.
The two 2015 correction descriptions name other crop/soil tables, not Cotton;
this narrows follow-up but does not certify the entire report or a release clock.

- Per-release actual publication clock and original/revised-version provenance.
  The stated after-4-p.m. policy is a lower bound, not proof of the exact clock.
- Accessible FAS original report files and reviewed units (running bales versus
  other measures); maintain marketing-year identity at overlapping boundaries.
- Broader reconciliation for years besides 2023 and missing-week diagnosis
  before compiling a package.
- AMS excerpt archive coverage, exact daily publication timing, correction
  policy and source-use review before any same-day feature is admitted.

The first pilot evidence and results live under `output/usda-vintage-pilot/` and
are copied to the user's Drive reports directory. They contain no credentials.
No new training, source-bundle replacement, or experiment identity change is needed
to inspect these diagnostic outputs.

## WASDE structured-report pilot (2026-09-28)

USDA's [historical data policy](https://www.usda.gov/historical-wasde-report-data-3)
explicitly describes as-published estimates, excluding subsequent underlying-series
revisions; individual reports remain the official record. This makes WASDE a
priority alongside the still-blocked NASS vintage review. It does not establish
an exact publication clock or exclude correction/reposting exceptions.

The consolidated CSV/ZIP endpoints returned HTTP 403 from the local environment.
The official ESMIS January 12, 2023 XML was accessible and archived instead:
`output/wasde-vintage/wasde/a87310fb5f934359831f218c486cdab1b97241415e9305c39dc32936db9769ff`.
`sources.wasde` extracts 588 World Cotton cells, retaining report month, marketing
year, forecast month, region, attribute and million-480-pound-bale units. Previous
and current forecast columns are not collapsed. Verified footnote `3/` means less
than 5,000 bales and stays a censored value, never an invented zero. Unknown units,
duplicate dimensions, unknown numeric markers and corrupt archives fail closed.

Run `python -m cottonlens_ml.sources.wasde --archive <archive-folder> --output <new-report.json>`.
This is an inspection report, not an admitted publication package. Pending:
cross-format numeric reconciliation, release/correction timing, historical schema
coverage and usage review. No publication timestamp is inferred from XML report
month or the ESMIS technical date field. No notebook/source bundle was replaced.

Cross-format check: December 9, 2022, January 12, 2023 and April 11, 2023 official
XML/TXT pairs each match all 196 checked cells (588 comparisons total). Scope is
World, United States, China, India, Pakistan, Brazil and Australia, across every
season/forecast-month column in those tables. This is parser consistency evidence,
not independent economic truth or first-release certification. Other regions are
extracted from XML but not covered by this TXT comparison. Use optional
`--text-archive <TXT-archive-folder>` to produce a checksummed comparison report;
month, unit, coverage and duplicate-key errors fail closed; numeric differences
are explicitly listed. Evidence lives in `output/wasde-vintage` and Drive reports.

May 12, 2022 season-transition check also matches 196/196 cells (four reports,
784 comparisons cumulatively). The new 2022/23 season's April forecast is `NA`,
while May has the initial estimates. Preserve `NA` as unavailable, distinct from
both numeric zero and censored `3/` values. Cross-format equality includes the
qualifier, so `NA` and less-than-5,000-bales cannot falsely match. A subsequent
revision feature must join the same marketing year; absence of a previous-season
estimate must not produce a zero-based or cross-season revision.

Batch status: `python -m cottonlens_ml.sources.wasde --audit-root output/wasde-vintage
--output <new-audit.json>` verifies all source hashes, follows archived page links
to the exact XML/TXT receipts, rejects ambiguous versions, and recomputes numeric
comparisons. The initial audit has 4/4 numeric passes, zero admitted releases.
Remaining exit gates are explicit: declared historical coverage, publication and
correction evidence, usage review, and compiled-package/notebook integration.
The four ESMIS pages contain `12:00:00Z`; retain that as a technical field rather
than treating it as noon Eastern or certifying first publication. This command
does not download data, train, or silently relax the existing admission policy.

2023 bulk acquisition now covers all 12 monthly releases discovered on official
ESMIS listing pages 3 and 4, with all 2,352 selected-region XML/TXT cells matching.
The immutable result is `output/wasde-vintage/year-2023-acquisition.json`.
`archive_release(page_url, root)` reuses checksum-verified receipts and rejects
ambiguous cached versions; it never refreshes or overwrites an archived version.
This closes 2023 acquisition/numeric coverage, not full 2010-2023 history or admission.

The [official errata list](https://www.usda.gov/historical-changes-revisions)
reviewed on 2026-09-28 identifies the May 12, 2022 repost as dairy discussion text,
with PDF correction at 2:25 pm. The August 2022 correction concerns sugar text.
These descriptions do not name Cotton table changes. Record correction scope
separately from report-level reposting; absence of a listed Cotton correction
alone does not certify first-publication timing or complete revision coverage.

2020-2022 bulk pass: all 36 release pages and XML/TXT pairs acquired from a
frozen listing-derived manifest (`manifest-2020-2022.json`). Eight May-December
2022 pairs pass numeric comparison. The preceding 28 releases are blocked by
the missing second TXT column-header line, not by demonstrated numeric mismatch.
January 2020 official PDF pages 26-27 contain the full column labels; PDF evidence
SHA256 `6822bd76d7016a3e7d9f7b986aac08e95532ae8d5fb67b25fec7205833eaed6f`.
The first PDF page was visually reviewed; the column order and seven selected-region
values match XML. The TXT header check remains strict.

The reusable acquisition command is `python -m cottonlens_ml.sources.wasde
--manifest <frozen-manifest.json> --archive-root output/wasde-vintage
--output <new-acquisition-report.json>`. It resumes verified downloads and records
each failed release explicitly. Current 2020-2023 raw coverage is 48 monthly
releases; 20 XML/TXT pairs pass, 28 require alternate-format verification. None
has been admitted to training by these checks. See `acquisition-2020-2022.json`.

PDF reconciliation of the older format raises combined numeric coverage to 46/48
for 2020-2023; see `numeric-evidence-2020-2023.json` in the local output and Drive.
The remaining exceptions are 2021-08-12 and 2021-09-10. PDF text extraction
overlaps some rows on those pages, so the automatic seven-region comparison fails.
Do not count either as verified merely because other rows look plausible.
Numeric agreement still does not verify first-release availability, later
corrections, source usage, or complete pre-2020 history. The research notebook
remains gated from using this source.

Follow-up on 2026-09-29: the two exception PDFs were visually inspected and
independently parsed with `pypdf` fixed-column layout extraction. Its full rows
for the seven named regions were compared to the archived XML, with exact
season/forecast-month coverage and column-order checks. All 196 cells in each
exception match. The immutable successor record is
`numeric-evidence-2020-2023-r2.json`: **48/48 monthly reports, 9,408 selected
cells** now pass numeric comparison (20 XML/TXT, 26 XML/PDF ordinary layout,
2 XML/PDF fixed-column layout). The original 46/48 record remains unchanged.
`ml/review_wasde_pdf_layout.py` reproduces the successor record from the frozen
PDF manifest, prior record and checksummed raw files; it requires `pypdf` only
for this offline evidence review. A changed or missing PDF/XML cell fails closed.

Availability is still blocked. ESMIS page markup such as the
[January 2023 release](https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/2023-01-12)
contains `12:00:00Z`, which is not 12:00 Eastern. USDA ERS describes current
WASDE releases as generally at [noon Eastern](https://ers.usda.gov/topics/farm-economy/commodity-outlook/usda-outlook-process),
while [earlier releases](https://www.ers.usda.gov/amber-waves/2012/june/wasde)
used 8:30 a.m. The ESMIS date field is therefore **not** proof of an actual
release clock. The [USDA correction list](https://www.usda.gov/historical-changes-revisions)
also records reposts. No timestamp is inferred from numeric equality, and
`model_eligible` remains false. Before admission, establish a documented
per-release availability/correction policy, source-use conditions, and the
pre-2020 history needed for the frozen research folds.

## Earlier WASDE history (2016–2019, 2026-09-29)

Official ESMIS listing pages 8–12 are checksummed in
`output/wasde-vintage/manifest-2016-2019.json`. They identify **47 published
months and 50 release-page links**. The missing January 2019 month is real:
the [February 8, 2019 USDA report](https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/2019-02-08)
states on its first page that the January report would not be published because
of the federal funding lapse. Its archived PDF SHA256 is
`ac793c2d7c5e72a53ced7880ca07a5b410d2c1963e3b61d010d545adfd58610c`.
An absent release is not backfilled with a fabricated January vintage.

Two months have extra archive links: December 2018 has December 11 and 14
versions, and November 2019 has three November 8 pages. All seven selected
regions' Cotton XML cells match between the December versions despite
different full XML file hashes. The November pages point to identical XML,
TXT and PDF bytes. The archive retains each page URL and immutable aliases
when different official file URLs share the same content hash. This establishes
Cotton numeric agreement across these copies, not which file first became
public or whether non-Cotton sections changed.

The first nine 2016 releases had no official TXT link. The original strict
acquisition report (`acquisition-2016-2019.json`) remains unchanged; its
successor (`acquisition-2016-2019-r2.json`) explicitly archives XML and PDF
for those months. No missing TXT was synthesized. An offline two-extractor
review (`ml/review_wasde_history.py`) compares the PDF table against XML,
including every season and forecast-month column. Five 2019 PDF text layers
attached the literal, invisible `filler` to complete cells; only this exact
suffix is stripped before the full numeric check. The immutable
`numeric-evidence-2016-2019.json` records **50/50 links and 9,800/9,800
selected cells passing**, with per-release checksummed checkpoints. There
were no numeric mismatches. Alongside the 2020–2023 review, numeric evidence
now covers 2016–2023, but the extra links must not be counted as independent
monthly observations.

The 2016–2019 raw pages, XML/TXT/PDF files, URL-alias receipts, checkpoints,
manifest, numeric report and exact review-code snapshots are bundled in
`wasde-2016-2019-evidence.zip` (SHA256
`ca95d8fa646e9cd3259d6b66dce632f79228d85f7078735f346bbd9d900ea815`).
The 35.4 MiB ZIP and its receipt were copied to the user's synced Drive folder
with matching checksums; the original local raw archives remain in place.

These passes **do not make WASDE model eligible**. Per-release actual
availability and first-versus-corrected publication evidence remain open,
as do source-use review and any pre-2016 history needed to train the first
research folds. No Colab notebook, training run, or publication package was
changed by this review.

## WASDE availability decision (2026-09-29)

The offline, immutable `availability-inventory-2016-2023-r2.json` joins the two
numeric reviews and verifies their declared release coverage. Across 96 calendar
months, 95 have an official release; January 2019 was cancelled. The 98 archived
page links include the extra December 2018 and November 2019 pages. All 98 have
passing selected-table numeric checks. **Zero months have a verified actual
first-publication clock or original-versus-repost provenance in this evidence
set.** This is an evidence-status count, not a claim that USDA lacks such records.
The report includes its review-code SHA256, is reproducible offline with
`ml/review_wasde_availability.py`, and
fails if a source review is incomplete or its coverage disagrees with the
manifest.

The [official ESMIS archive](https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates)
labels release *dates*, while [USDA ERS](https://ers.usda.gov/topics/farm-economy/commodity-outlook/usda-outlook-process)
describes the general noon-Eastern schedule. Neither certifies the actual first
availability of each archived file. The [USDA changes/revisions list](https://www.usda.gov/historical-changes-revisions)
shows that report corrections and reposts occur; no notice for a Cotton cell is
not proof that every archived XML is the original version. An ESMIS page such as
the [May 2026 repost](https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/2026-05-12-0)
explicitly distinguishes a later version. Accordingly WASDE remains a research
archive, **not a model feature**. More GPU runs cannot resolve this timing and
vintage gap. Next source priority is a report archive with independently
verifiable per-report publication and correction metadata, beginning with the
AMS `MP_CN001` report index; its public search timed out in bounded local probes,
so no timestamp has yet been certified there either.

## NASS Crop Progress content history (2010–2023, 2026-09-29)

`ml/prepare_nass_history.py` selects the exact SHA256-checked annual national
Cotton Quick Stats snapshots in `output/reports/usda-history-acquisition-check.json`,
then compares each Upland condition week to the original ESMIS Crop Progress
TXT linked by its dated release page. Raw page/TXT bytes, receipts and immutable
year audits are under `output/usda-nass-history-2010-2023/`. The audit covers
**311 observation weeks**: **308 complete five-category numeric matches** and
**3 weeks where the official report says 0% but the current API omits that
category**. The latter are marked `reference_missing_zero`, not called full
independent matches. Both 2012-10-29 and 2018-09-10 pages carried official
delay notices; the notices are archived separately, and the subsequently
released weekly reports were checked. Two 2011 links to one report were
accepted only after their downloaded bytes matched exactly.

`ml/export_nass_diagnostic.py` produces the 311-row
`upland-condition-2010-2023.csv` (SHA256
`733293135b7e34ec1d09c355538074338e9abc36deba84ae2905d5ffab4f33a5`)
and frozen manifest after rechecking all archived bytes. The private evidence
ZIP contains 989 checksummed members, including the selected original API
snapshots, dated report pages, TXT files, delay notices, year audits and review
code. ZIP SHA256:
`825b1bbfc59ebe96d1612b37e547a107533deda7ebc4f6d171cd1e82bdc6d62b`.
The CSV, manifest, ZIP and receipt were copied with matching checksums to the
user's synced Drive `reports/nass-diagnostic` folder.

This is a **content archive only**. The current Quick Stats snapshot is not
evidence of its original historical state, and the ESMIS page's date field
does not establish actual first availability or correction provenance.
`model_eligible=false` remains in the manifest. No NASS values were added to
features, Colab training, or the locked benchmark.

## AMS archive transport correction (2026-10-01)

Ordinary `requests` timed out on the publication listing both locally and in
Colab. A browser rendering the page was insufficient evidence that the notebook
transport worked. The same listing and a dated TXT now pass live acquisition
through locked `curl-cffi==0.13.0` with the `chrome136` transport, TLS verification
enabled and environment proxies disabled. The first page parses 50 records; the
2023-07-31 TXT matches the frozen ESMIS SHA256 exactly. This establishes an
operational alternative, not the cause of USDA's server behavior.

The optional `data` environment is selected by the existing Data Workbench;
research notebooks retain their original immutable bundles. Only interrupted
network transfers receive up to three bounded attempts. HTTP denial, malformed
metadata, changed filters and corrupt cache files remain errors. Partial streams
produce no completed request or review record; verified downloads are retained.
A long-lived client later failed on a TXT after more than 200 successful
downloads; a fresh client retrieved the exact same URL. Interrupted transfers
therefore also renew the client before retrying, and all clients/streams are
closed. This is recovery evidence, not a certified explanation of server limits.
The complete 40-page index includes dated legacy `MP_CN001YYYYMMDD.TXT` names;
these are accepted only when their embedded date and versioned path agree.
Publication timezone, first-version provenance and model eligibility are
unchanged by this transport fix.

## AMS completed availability review (2026-10-01)

The complete 40-page MMN content review matches 998 of the 999 frozen
2020–2023 ESMIS reports byte-for-byte. The missing metadata date is 2020-01-17;
its ESMIS report exists, but supplementary exact-date searches under slug 3005
and MP_CN001 did not recover an index row. Absence in these searches is not
proof that no other archive contains it.

`ml/review_ams_availability.py` reproduces an offline, checksum-verified audit.
Among the 998 Final rows, **13 displayed publication dates follow report dates
by 1–8 calendar days**, with unverified cause, and **21 filenames have numeric
suffix flags**. Neither Final nor a filename suffix establishes original-versus-
corrected status. The captured public correction index has 114 distinct slugs
and no 3004 row; it does not certify a complete historical correction ledger.

One separate 2025-07-31 record has an official public-index UTC epoch and exact
canonical/MMN payload match. Its UTC clock is 18:45:31.803 and the MMN display is
13:45:31. This single observed offset is not applied to 2020–2023, is outside the
selection cohort, and does not certify a first version. Historical verified UTC
count remains **zero**, so **model_eligible=false**. A corrected version can be
used only after its own actual availability is established; obtaining an
original version is not the only possible admissible route.

Audit ID: `80f5257659a93e344ce9a98c6ebb81fe646e5fc26d4aebfdd21774e97ba8586a`.
Raw responses, audit, decision and unsent publisher clarification draft are in
`output/ams-availability-review-20261001` and private Drive
`reports/ams-availability-review-20261001`. This closes the current evidence
review, not the source's model-admission gate. Next step is authoritative
historical metadata semantics and per-version clocks, rather than repeating
the unchanged-data model search. Existing notebooks and immutable bundles are
preserved.

## API route closure and observed snapshots (2026-10-01)

The official API corrections guide defines `listPublishedReports/all` as the
most current publication date of each report, not all its historical versions.
The July 2025 Cotton FAQ explicitly excludes pre-August-2025 numeric price data
from the API and points to the historical report archive. The accompanying
Cotton search guide documents that archive but does not resolve its historical
clock timezone or per-version timing. These three guides and the route decision
are frozen under `output/ams-api-route-review-20261001`. Increasing a `lastDays`
window or repeating the modern numeric query cannot certify the missing history.

To avoid recreating the same evidence gap for future data,
`cottonlens_ml.sources.public observe` now makes a fresh public request and
records exact source bytes with UTC request-start and download-completion
timestamps. Same bytes may share their content-addressed raw file; each actual
observation has its own immutable receipt and preserves its URL. Partial
transfers, invalid/backwards clocks, changed bytes and future cutoff requests
are rejected. Retrieval/HTTP dates are never renamed first-publication times.

The existing Data Workbench has an optional CPU-only `observe_ams` action for
the current public publication index and `ams_3804.pdf`; default remains
`inventory`. A first real capture was verified on 2026-10-01 and copied to
private Drive `data/observed-public-sources-v1`. These are **observed snapshots,
not admitted features**. Historical compilation, cohort and model gates are
unchanged; `model_eligible=false` remains explicit. They establish that these
bytes were available by our actual download completion, not that they were
original or available in the 2016–2023 research period. There is no background
scheduler: another observation occurs only when this action runs.

## FAS ESRQS archive identity review (2026-10-01)

The public ESRQS portal documents an Archived Weekly Reports view. Its 2020
index contains 52 rows, all with sentinel `createdTime=0001-01-01T00:00:00`;
this is not a publication clock. The portal's own public-client request route
was checked before interpreting the returned files.

Requests for archive IDs `2248fde3-edca-41ba-99a5-0274a21e9a7c`
(index label 2020-06-04) and `6b702a17-3665-4769-afd3-36decce617ad`
(2020-05-28) returned identical PDF bytes, SHA256
`aa7e8d16af70a889edb661297a2ff484713eca9c2288b57d4973e857cf7784b8`.
The PDF declares a reporting period ending **2026-09-24**. Quarantine both
responses: they cannot reconcile historical 2020 API numbers or enter a
backtest. The cause of this response mismatch is unverified; neither valid
HTTP/PDF content nor an archive listing proves historical identity.

`ml/review_fas_archive.py` reproduces this offline from the checksummed raw
index and PDF retrieval receipts using pypdf. The period check deliberately
does not assume the index date labels the reporting period instead of the
release. Even matching dates never grant publication/vintage verification.
The canonical review is
`output/fas-esrqs-probe-20261001/archive-identity-review-v2.json`.
The earlier frozen review is preserved, not overwritten.

`output/reports/free-source-readiness-20261001.json` summarizes the six
reviewed source channels and pins their audit records. **Zero new external
feature groups are admitted.** Market-only research remains the existing
CT=F/DXY/WTI cohort; USDA/ECB raw/content archives remain outside its model
inputs. Climate, satellite and text/event packages are not admitted either.
This summary does not repeat full raw-archive checks or claim improved
prediction performance. Next dependency: correct FAS archived payloads and
authoritative date/version semantics, not another unchanged-data GPU search.

## FAS historical response recovery (2026-10-01)

The earlier mismatched responses remain quarantined. A later fresh anonymous
session with `Cache-Control: no-cache` recovered two different PDFs whose
declared reporting periods correctly end 2020-05-28 and 2020-06-04. Responses
advertise `max-age=600`. This observation does **not** isolate the original
cause: session renewal, elapsed time and request headers changed. Do not call
it a proven cache-key bug or silently replace the earlier evidence.

The recovered legacy CAM format lacks the modern ESR title; the period parser
now accepts its explicit period sentence. `ml/review_fas_archive.py` compares
only the ALL UPLAND COTTON table, not Pima or another cotton class. The pinned
commodity catalog identifies code 1404 as All Upland Cotton with unit ID 2;
its PDF table prints thousands of running bales. Across the two selected weeks,
API sums for outstanding sales, accumulated exports and next-year outstanding
sales match all **six** printed totals within the 50-running-bale rounding
half-width. This proves sampled content agreement at printed precision only,
not whole-history consistency, original vintage or availability.

A separate USDAFAS GovDelivery attachment for the 2020-06-04 period prints an
embargo of **08:30 AM on 2020-06-11**. Its cover has no explicit timezone and
the declaration is planned release timing, not observed public release.
The attachment's path date, PDF creation time and today's retrieval/HTTP dates
are not substituted for actual historical availability. The attachment has a
different layout; no numerical comparison or same-version certification is
claimed for it. Source:
https://content.govdelivery.com/attachments/USDAFAS/2020/06/10/file_attachments/1471324/wr06042020.pdf

Canonical recovery record:
`output/fas-cache-control-probe-20261001/content-recovery-review-v2.json`.
The updated six-channel state is
`output/reports/free-source-readiness-20261001-v3.json`; the older states and
24-file Drive identity-review delivery are unchanged. Model admission remains
false. The next dependency is a timestamped official dissemination record
bound to the actual version, not further unchanged-data training or another
full content-archive scan. No publisher message has been sent.
