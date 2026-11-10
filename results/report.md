# jikan results

Synthetic churn-style data (4000 users, 20% churn, one logistic-regression feature = ticket count), 3 seeds, train on first half of users, test on second half. `offline` = AUC on test rows joined the same way as training; `production` = AUC when the same model is fed correct point-in-time values (what serving would see).

## Leaky joins vs point-in-time join

| future-only extra tickets for churners | model trained on | offline AUC | production AUC | rows that used a future value |
|---|---|---|---|---|
| 0 | latest value (bug) | 0.607 | 0.609 | 100.0% |
| 0 | same-day join (bug) | 0.609 | 0.609 | 21.4% |
| 0 | point-in-time | 0.609 | 0.609 | 0.0% |
| 1 | latest value (bug) | 0.810 | 0.609 | 100.0% |
| 1 | same-day join (bug) | 0.652 | 0.609 | 21.4% |
| 1 | point-in-time | 0.609 | 0.609 | 0.0% |
| 3 | latest value (bug) | 0.974 | 0.609 | 100.0% |
| 3 | same-day join (bug) | 0.686 | 0.609 | 21.4% |
| 3 | point-in-time | 0.609 | 0.609 | 0.0% |

## Ignoring ingestion delay (values reach the store 2 days after they are written; churners spike in the 2 days before the event)

Join ignoring the delay: offline AUC 0.854, production AUC 0.609; audit says 63.6% of rows used a value not yet available at the event.

## Speed (1M events, 5M feature rows, 100k entities, 1 thread)

- jikan `asof_index`: 1.31 s
- pandas `merge_asof`: 0.54 s (sorting both inputs included, as in jikan)
- agreement on matched timestamp: 100.0000%