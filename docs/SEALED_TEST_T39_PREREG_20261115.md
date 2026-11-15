# T39: one-shot sealed test of T36 on 2022–2024 (PRE-DECLARED 15 Nov 2026)

On 15 Nov 2026 the user approved spending the sealed years on T36 ("Evet, mührü aç", then "Evet, devam"). Earlier, the sealed shot had been reserved for the chain's final claim. No 2022–2025 value had been read when this document was written. Only metadata was checked: release dates and data-catalogue coverage.

Code frozen at commit `fe716ac`, plus the sealed-data pull and feature-extension changes listed under "Allowed changes".

## Primary claim
- **T36 at 1 Oct, unchanged.**
  - Members: B (T25), H (T06 HGB with `fsa_aug` area), N (T35 CNN seed mean) and U (raw USDA October).
  - Weights for year Y come from the 0.05 simplex fitted on 2013…Y−1, using out-of-sample member predictions only. So weights for 2022 use 2013–2021, and weights for 2024 also use the 2022 and 2023 out-of-sample predictions.
  - Fallbacks, exactly as coded: B when fewer than 2 past years exist; B for a state that lacks any member.
- **Years:** 2022, 2023, 2024. USDA published no October 2025 forecast (no Crop Production release in October 2025), so 2025 cannot enter the 1 Oct test.
- **Pass rule (T36's, unchanged), over 2022–2024.** All of the following must hold:
  - TX and US MAPE below USDA October's;
  - TX and US MAPE below T25 on the same rows;
  - production-weighted state error below both USDA October's and T25's.
- **Reported alongside:** per-year errors of T36, T25 and USDA, per-year member coverage (how many states fell back to B), and the weights used.

## Secondaries (reported, never claimed): all become visible once the finals are pulled
- T25 alone vs USDA October, 2022–2024.
- H alone (T06 with `fsa_aug` area) vs USDA October.
- T36 at 1 Sep (B+H+U) and at 1 Aug (B+H+N+U), years 2022–2025 where USDA's forecast and the inputs exist. For 2025, MODIS on Planetary Computer has no composites after June, so N and H fall back as coded.
- T37 and T38 (Nov/Dec) vs USDA December, 2022–2025 where data exist.

## Allowed changes (data plumbing only, no modelling change)
- **NASS finals and forecasts for 2022–2025.** Pulled into new files and merged into copies. Dev rows (≤ 2021) are checked to be identical to the dev pull, and differences are reported rather than adopted.
- **FSA crop acreage files for 2022–2024,** parsed by the same `fsa_acreage.py`. Unknown layouts are skipped as coded, which makes the area fall back to June Acreage; this is reported.
- **June Acreage, POWER weather, Drought Monitor:** extended by the same scripts.
- **Cotton mask: a declared source change.** Planetary Computer's `usda-cdl` ends in 2021. The prior-year CDL for 2023 and 2024 (CDL 2022 and 2023) comes from NASS CropScape, as state GeoTIFFs of the same CDL product (class 2 = cotton), read at the same reduced resolution. The 2022 season uses CDL 2021 from Planetary Computer, as before.
- **MODIS MOD/MYD13Q1 v061 from Planetary Computer,** as before.
- **CNN:** histograms for 2022–2024 come from the same `county_hist.py` with the mask change above. `train_cnn.py` is unchanged, with test years extended to 2024; each fold trains on years < Y−1 and validates on Y−1.

## Limits, stated before the result
- Three years cannot establish statistical significance. Even 3 of 3 wins has about a 1/8 chance under a coin-flip null.
- A loss is informative. A win is encouraging only.
- Report events, prices and returns stay sealed. This test unseals production finals and USDA forecasts only.
- Execution: one driver run. Outputs are committed whatever they show, and nothing is re-run after the finals are seen, except to fix a crash, which would be disclosed.

## Push
The repo rule asks for a pushed preregistration before unsealing. Pushing needs the user's explicit approval. If the user declines, this local commit is the record and the waiver is noted in the ledger.

## Amendments before any outcome was read (15 Nov, evening)
- **FSA snapshot rule.** The parser had a literal rule that skipped labels containing "August 22", written for 2019's duplicate August file. It would have dropped 2022's only August file (as of 22 Aug 2022).
  - Replaced by the rule the code comment states: keep the earliest snapshot per year and month.
  - The dev FSA csv is byte-identical after the change.
- **Coverage known so far (inputs only):**
  - FSA files: 2022 Aug/Sep/Oct, 2023 Aug/Sep/Oct, 2024 Aug/Sep. There is no 2024 Oct file, so 2024 has no Oct `fsa_gap`. That affects only B's training rows for later years.
  - June Acreage 2022–2025: all parsed.
  - POWER 2022–2024: pulled.
- **What was seen, nothing else.** Row counts of the NASS sealed pull and the FSA state totals of planted/failed acres (inputs). No production final, USDA forecast value or error was printed or read.
- **The features builder** now reads every weather span file and takes years from `COTTONLENS_LAST_YEAR`. The dev oct1 features rebuild matches the stored file (max difference 5e-13).
