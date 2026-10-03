# Day 1 R2 evidence review

Release: `v20260924-185501-519b9c1a`. ZIP SHA-256:
`938c2f119ec417549f37bbf7c5b226cde66be18deeb89b1b927292be041657f6`.
The release checksum, matching runtime validation receipt, frozen identities,
raw market and processed feature snapshot checksums were verified read-only.
No model was fitted and no existing release or snapshot was modified.

## Missing LSTM curve

The selected development fold 4 trial (64 units, dropout 0.2) has 13 recorded
epochs; best validation loss is 0.6204919815063477 at epoch 3. Its model and
history checksums match the completed checkpoint in readiness directory
`e153054a49965a3e176aedc4105280232dcda62ac81cfac5c9889cd31bb92b82`:
`walkforward-fold-4/lstm-95a5b4c00fe9c74d41863a2c5e4421ee5156ce4d44d139fdf5203546acf302a9.history.json`.

The audit refit intentionally clears the candidate training history, but the
report previously read that cleared field and printed zero epochs. The source
now retains selected validation curves in each development fold and identifies
their provenance in the report. Older releases without embedded curves report
unavailable evidence rather than zero training. Existing R2 evidence remains
unchanged; this review supplements it. These are validation trial curves, not
curves from audit or deployment refits.

## Volume availability

The frozen raw snapshot contains 4,205 Cotton observations: 234 zero-volume
rows, no null or negative volumes. All 235 missing `cotton_volume_change`
values exactly match the current formula: one initial observation and 234
observations with a zero previous-volume denominator. The source of vendor
zero-volume values is unverified; genuine zero trading cannot be distinguished
from provider/proxy artifacts using these snapshots alone.

No imputation, feature change, cohort replacement or performance tuning was
performed. The 504-origin complete-feature benchmark is conditional on data
availability and cannot establish performance on excluded observations.

## Outcome and remaining boundaries

Both horizons retain Naive under the fixed selection gates. The already seen
historical audit remains descriptive and cannot direct tuning. V1 exact
replication and recovery of the prior local backend database remain unresolved.
The reporting correction does not justify rerunning training or changing the
frozen R2 experiment. Fourteen focused report, walk-forward orchestration and
pipeline tests passed, together with lint on changed source and tests.
