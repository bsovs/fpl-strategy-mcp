# Observed manager action heads

The project now has a separate reverse-engineering audit for observed manager
behavior. It is deliberately kept outside the production policy because the
only exact weekly manager archive currently available locally is 2025/26, the
untouched final test season.

Run it with:

```sh
PYTHONPATH=src python scripts/reverse_engineer_observed_actions.py \
  --history-root /path/to/fpl-history \
  --output runs/elite-manager-benchmark/action-heads.json
```

## What is modeled

The script fits separate, group-cross-validated logistic heads for:

1. transfer versus hold;
2. a multi-transfer bundle versus a single transfer or hold;
3. accepting a paid hit versus a free transfer or hold; and
4. using any chip versus holding the chip.

It also fits an incoming-versus-outgoing player signal head using lagged
points, minutes, price, ownership, and transfer momentum. That last head is an
observed behavior profile, not a legal recommendation model: it does not yet
have every affordable, position-compatible alternative in the choice set.

The features are cut at the previous gameweek. They include form and minutes
windows, price, selected count, transfer momentum, season phase, prior bank and
squad value, and the previous event's points/rank. Realized future points,
final points, final rank, and the current action are labels or audit metadata,
never features.

## 2025/26 holdout audit

The archive contains 100 managers, 3,700 post-opening decisions, 2,131
transfer weeks, 1,048 multi-transfer weeks, 177 paid-hit weeks, and 784 chip
weeks. Grouped by manager, the descriptive heads achieved:

| Head | Group-CV AUC | Interpretation |
|---|---:|---|
| Transfer vs hold | 0.670 | Season phase, cumulative points, and squad form explain some transfer timing. |
| Bundle vs single/hold | 0.736 | Bundle behavior is distinct and should not be modeled as repeated 1:1 swaps. |
| Paid hit vs free/hold | 0.623 | Hit decisions are sparse and weakly identified from the current state alone. |
| Chip vs no chip | 0.687 | Chip timing is a separate state/resource decision. |
| Incoming vs outgoing profile | 0.590 | Current lagged player signals alone do not explain the full choice set. |

These are behavior-prediction metrics, not claims that the heads cause rank or
points. The coefficients and the leakage contract are saved in
`runs/elite-manager-benchmark/action-heads.json`.

## What this says about the current gap

The evidence supports a multi-head strategy: a transfer-value model should
first estimate player and price outcomes; a hold/transfer head should decide
whether the improvement is large enough to spend a transfer; a bundle/hit head
should price the marginal transfer cost; and a chip head should reserve scarce
resources for the right fixture state. A single player-points ranking cannot
learn all of those decisions reliably.

The present archive still has two material gaps:

- earlier seasons do not yet have complete point-in-time manager squads and
  actions, so the heads cannot be trained without using the 2025/26 holdout;
- the archived weekly picks do not include each player's purchase price and
  selling value, so exact affordability and price-loss logic cannot yet be
  reconstructed for every observed move.

The next valid training input is a pre-2025/26 weekly manager archive with
the squad, purchase/selling prices, bank, free transfers, chips, deadline
timestamp, and available alternative pool at each decision. Until then, the
heads are diagnostics. The corrected baseline replay remains below the
2,413-point target; the separate external-style forecast candidate is reported
in the main data-quality audit and is not trained on these holdout actions.

## External challenger

The public [fpl-luck-or-skill repository](https://github.com/zakariae-boui/fpl-luck-or-skill)
reports a 2,431-point patient-chip backtest. Its published rule is a useful
challenger: avoid paid hits, compare multi-week expected value, use chips
before their half-season deadlines, and optimize a legal squad globally. The
claim is externally reported rather than independently reproduced here: the
public raw prediction artifact is not included and the LightGBM/OpenMP runtime
did not run cleanly on this Mac. The local simulator now includes a
holdout-safe `patient_chips` challenger so the rule can be compared without
promoting the external result to ground truth. The Mac-compatible
`external_hgb` forecast path now produces a 2,486-point 2025/26 replay when
paired with a forecast-optimized opening and the patient-chip rule; that is a
promising research candidate, not a causal validation of the observed heads.
