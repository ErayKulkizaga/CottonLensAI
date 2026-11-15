# Link 1 preregistration (DRAFT, waiting for the user's approval): report-day price reaction

Written before any report-day return was computed. This draft sets a new success criterion, so it needs the user's approval before the test runs.

## Question
1. Does the cotton price move more on USDA Crop Production report days than on other days?
2. Does the direction of the move follow USDA's revision of US production?

If both hold, predicting the revision (Link 2) has a price payoff.

## Data (2012–2021 only; the seal also covers 2022–2025 events)

- **Events:**
  - **Primary set:** US Crop Production reports in September, October, November and December of 2012–2021. Each has a previous-month forecast to compare against. Oct 2013 is missing (shutdown, no report), so about 39 events are expected.
  - August is excluded from the primary set because it has no previous survey forecast. January (the final figure) is also excluded.
- **Release dates:** taken from the ESMIS Crop Production listing. The report is released at 12:00 ET, while ICE cotton is trading. If the release falls on a non-trading day, the next trading day is used.
- **Revision:** r = log(USDA_m / USDA_m−1) for US all-cotton production, from the NASS files already pulled (≤2021).
- **Price:** the repo's daily `cotton` close series (`market_history.parquet`), filtered to ≤2021 before anything is printed.
- **Event return:** R = log(close_t / close_t−1), where t is the release trading day.
- **Non-event days:** all trading days from August through January of the same years, excluding event days and the day after each event.
- **Flags:**
  - Days with |R| ≥ 4% (a possible limit move) are reported with and without.
  - Roll days cannot be identified in this continuous series. This is recorded as a limitation.
- **Confounding:** WASDE (world supply and demand) is usually released the same day. The event return therefore also carries world-balance news.

## Hypotheses

Each is tested at α = 0.025, one-sided, so that both together have α = 0.05.

**H1, reaction size.** Mean |R| on event days divided by mean |R| on non-event days is greater than 1.
- Test: permutation test. Draw 39 non-event days at random from the same months, 10,000 times, seed 42.

**H2, sign follows revision.** Spearman ρ(r, R) < 0, because higher production should mean a lower price.
- Test: permutation test, 10,000 shuffles, seed 42.
- With 39 events, about |ρ| ≥ 0.32 is needed.

**Practical gate, in the existing protocol's terms.** An *oracle* rule R̂ = β·r, with β fitted only on events of earlier years, the first two years used as warm-up.
- Compare its event-day MAE with the naive forecast R̂ = 0.
- The pass bar is a gain of at least **5%**.
- This rule knows the true revision, so it is an upper bound on what Link 2 can deliver for price.

## What each outcome means

| Outcome | Meaning | Next step |
|---|---|---|
| H1 and H2 pass, and the oracle gain is ≥ 5% | Predicting the revision would pay off in price | Link 2 (predict the revision) |
| H1 passes, H2 fails | The price reacts, but the month-to-month revision is the wrong measure of surprise | Link 3 (consensus) becomes necessary |
| H1 fails | The price does not react specially to these reports | Tell the user. The chain has no price payoff; work returns to production accuracy alone |

## Limits
- This is an association over about 39 events, not a causal estimate.
- The release clock is the scheduled 12:00 ET, not an observed first-access time.
- Results on 2012–2021 are development evidence, not a sealed test.
