"""python -m jikan.experiment -> results/report.md. Synthetic churn-like data; see README for what it can and cannot show."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .join import asof_index, day_granularity_index, latest_index, leakage_audit, take

DAY = 86400


def make(n_users, leak, seed):
    """Each user has one observation at t0 (days 30-60). Feature = support tickets, updated every ~2 days.
    Baseline signal: churners have a slightly higher ticket rate before t0. `leak` = extra tickets that appear
    AFTER t0 only for churners (a consequence of the outcome, as in real churn data)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n_users) < 0.2).astype(int)
    t0 = (rng.integers(30, 60, n_users) * DAY + rng.integers(0, DAY, n_users)).astype(np.int64)
    ents, ts, vals = [], [], []
    for u in range(n_users):
        n = 45
        t = np.sort(rng.integers(0, 90 * DAY, n)).astype(np.int64)
        rate = 1.0 + 0.5 * y[u] + rng.normal(0, 0.5)
        v = rng.poisson(max(rate, 0.1), n).astype(float)
        v += leak * y[u] * (t > t0[u])
        ents.append(np.full(n, u)); ts.append(t); vals.append(v)
    return y, t0, np.concatenate(ents), np.concatenate(ts), np.concatenate(vals)


def auc(model, X, y): return roc_auc_score(y, model.predict_proba(X)[:, 1])


def study(leak, seeds=(0, 1, 2), n_users=4000):
    rows = []
    for s in seeds:
        y, t0, fe, ft, fv = make(n_users, leak, s)
        users = np.arange(n_users)
        tr, te = users < n_users // 2, users >= n_users // 2
        joins = {"latest value (bug)": latest_index(users, fe, ft),
                 "same-day join (bug)": day_granularity_index(users, t0, fe, ft),
                 "point-in-time": asof_index(users, t0, fe, ft)}
        X = {k: take(fv, idx, 0.0).reshape(-1, 1) for k, idx in joins.items()}
        correct = X["point-in-time"]
        for k in joins:
            m = LogisticRegression().fit(X[k][tr], y[tr])
            rows.append(dict(leak=leak, trained_on=k, offline=auc(m, X[k][te], y[te]), production=auc(m, correct[te], y[te]),
                             seed=s, leaked=leakage_audit(joins[k], ft, t0)["future_written"]))
    return rows


def delay_study(seeds=(0, 1, 2), n_users=4000, delay=2 * DAY):
    """Features are only available `delay` after they are written. Joining as if there were no delay
    leaks values that serving would not have had."""
    rows = []
    for s in seeds:
        y, t0, fe, ft, fv = make(n_users, 0.0, s)
        # make values depend on recency of the outcome: churners' tickets spike in the 2 days before t0
        spike = np.isin(fe, np.where(y == 1)[0]) & (ft > t0[fe] - delay) & (ft <= t0[fe])
        fv = fv + 3.0 * spike
        users = np.arange(n_users); tr, te = users < n_users // 2, users >= n_users // 2
        naive = take(fv, asof_index(users, t0, fe, ft, delay=0), 0.0).reshape(-1, 1)
        right = take(fv, asof_index(users, t0, fe, ft, delay=delay), 0.0).reshape(-1, 1)
        m = LogisticRegression().fit(naive[tr], y[tr])
        rows.append(dict(offline=auc(m, naive[te], y[te]), production=auc(m, right[te], y[te]),
                         audit=leakage_audit(asof_index(users, t0, fe, ft, delay=0), ft, t0, delay=delay)["not_yet_available"]))
    return rows


def speed(n_events=1_000_000, n_feat=5_000_000, n_ent=100_000):
    rng = np.random.default_rng(0)
    fe = rng.integers(0, n_ent, n_feat); ft = rng.integers(0, 10**7, n_feat).astype(np.int64)
    ee = rng.integers(0, n_ent, n_events); et = rng.integers(0, 10**7, n_events).astype(np.int64)
    t = time.perf_counter(); idx = asof_index(ee, et, fe, ft); ours = time.perf_counter() - t
    left0 = pd.DataFrame({"e": ee, "t": et}); right0 = pd.DataFrame({"e": fe, "t": ft, "i": np.arange(n_feat)})
    t = time.perf_counter()
    left = left0.sort_values("t"); right = right0.sort_values("t")      # merge_asof needs sorted input: count it
    m = pd.merge_asof(left, right, on="t", by="e"); pdt = time.perf_counter() - t
    # agreement on the matched row's timestamp (ties can pick different but equal-time rows)
    got = np.where(idx[left.index.values] >= 0, ft[np.maximum(idx[left.index.values], 0)], -1)
    ref = np.where(m["i"].isna(), -1, ft[np.nan_to_num(m["i"].values, nan=0).astype(int)])
    return ours, pdt, float((got == ref).mean())


def main():
    out = Path("results"); out.mkdir(exist_ok=True)
    L = ["# jikan results\n", "Synthetic churn-style data (4000 users, 20% churn, one logistic-regression feature = ticket count), 3 seeds, "
         "train on first half of users, test on second half. `offline` = AUC on test rows joined the same way as training; "
         "`production` = AUC when the same model is fed correct point-in-time values (what serving would see).\n",
         "## Leaky joins vs point-in-time join\n", "| future-only extra tickets for churners | model trained on | offline AUC | production AUC | rows that used a future value |",
         "|---|---|---|---|---|"]
    for leak in (0.0, 1.0, 3.0):
        rows = study(leak)
        for k in ("latest value (bug)", "same-day join (bug)", "point-in-time"):
            r = [x for x in rows if x["trained_on"] == k]
            L.append(f"| {leak:g} | {k} | {np.mean([x['offline'] for x in r]):.3f} | {np.mean([x['production'] for x in r]):.3f} | {np.mean([x['leaked'] for x in r]):.1%} |")
    d = delay_study()
    L += ["\n## Ignoring ingestion delay (values reach the store 2 days after they are written; churners spike in the 2 days before the event)\n",
          f"Join ignoring the delay: offline AUC {np.mean([x['offline'] for x in d]):.3f}, production AUC {np.mean([x['production'] for x in d]):.3f}; "
          f"audit says {np.mean([x['audit'] for x in d]):.1%} of rows used a value not yet available at the event.\n"]
    ours, pdt, agree = speed()
    L += ["## Speed (1M events, 5M feature rows, 100k entities, 1 thread)\n", f"- jikan `asof_index`: {ours:.2f} s", f"- pandas `merge_asof`: {pdt:.2f} s (sorting both inputs included, as in jikan)",
          f"- agreement on matched timestamp: {agree:.4%}"]
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
