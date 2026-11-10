# jikan — 時間

**時間 (jikan)** means "time". Training sets for models like churn or fraud are built by joining feature values onto event timestamps. If the join uses a value written after the event, offline scores look great and production disappoints. This project implements a **point-in-time (as-of) join** with ingestion delay and TTL, two common *buggy* joins, and an audit that detects leaking rows, then measures what the bugs do. Feast and other feature stores implement this too; this is a small, checkable version.

## Results (`results/report.md`; synthetic churn-style data, 4000 users, 3 seeds)
Model: logistic regression on one feature (ticket count). "Production AUC" feeds the same model the correct point-in-time values.

| extra tickets churners get only after the event | trained on | offline AUC | production AUC |
|---|---|---|---|
| 0 (no leak) | any join | 0.607-0.609 | 0.609 |
| 1 | latest value (bug) | **0.810** | 0.609 |
| 3 | latest value (bug) | **0.974** | 0.609 |
| 3 | same-day join (bug) | 0.686 | 0.609 |
| any | point-in-time | 0.609 | 0.609 |

- The audit flagged 100% of "latest value" rows and 21.4% of same-day-join rows as using a future write, and 0% of point-in-time rows.
- **Ingestion delay:** with values reaching the store 2 days after being written, a join that ignores the delay gave offline AUC 0.854 vs production 0.609; the audit found 63.6% of rows used a value not yet available.
- **Correctness:** matches a brute-force reference on 30 random configurations (delay, TTL, strict/non-strict boundary) and `pandas.merge_asof` on random data; hand-written tests cover boundaries, unknown entities, TTL and delay.
- **Speed:** on 1M events x 5M feature rows, jikan took 1.31 s vs 0.54 s for `merge_asof` (both counting sorting), so it is not faster than pandas; it also returns indices (so the audit can run) and supports delay and TTL.

## Caveats
The data is synthetic and the leak is built in; this shows the mechanism and the size of the effect for a chosen strength, not how common it is. Timestamps must be integers. Ties between equal-time writes may resolve to different rows (tests compare timestamps). No streaming/online-store logic, no multi-feature-view handling, no comparison with Feast or Tecton.

## Run
```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]' pandas
.venv/bin/python -m pytest -q             # 8 tests
.venv/bin/python -m jikan.experiment      # ~10 s
```
