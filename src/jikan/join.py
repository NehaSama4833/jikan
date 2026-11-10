"""Point-in-time joins of feature updates onto events. Timestamps are int64 (e.g. seconds).

A feature value written at `feat_ts` becomes available at `feat_ts + delay`. An event at time `t` may use
a value only if it was available by then (<= t, or < t with allow_equal=False) and, if `ttl` is given, not
older than `ttl`. The result is the LATEST such value per event; entities with none get index -1.
"""
from __future__ import annotations
import numpy as np


def asof_index(ev_entity, ev_ts, feat_entity, feat_ts, *, delay=0, ttl=None, allow_equal=True):
    """Index into the feature arrays of the value each event may use, or -1."""
    ev_entity, ev_ts = np.asarray(ev_entity), np.asarray(ev_ts, dtype=np.int64)
    feat_entity, feat_ts = np.asarray(feat_entity), np.asarray(feat_ts, dtype=np.int64)
    avail = feat_ts + np.int64(delay)
    uniq = np.unique(feat_entity)
    f_rank = np.searchsorted(uniq, feat_entity)
    e_rank = np.searchsorted(uniq, ev_entity)
    e_rank_c = np.minimum(e_rank, len(uniq) - 1)
    e_known = uniq[e_rank_c] == ev_entity
    tmin = min(avail.min(), ev_ts.min()); span = int(max(avail.max(), ev_ts.max()) - tmin + 2)
    f_key = f_rank.astype(np.int64) * span + (avail - tmin)
    order = np.argsort(f_key, kind="stable")
    f_sorted = f_key[order]
    e_key = e_rank_c.astype(np.int64) * span + (ev_ts - tmin)
    pos = np.searchsorted(f_sorted, e_key, side="right" if allow_equal else "left") - 1
    ok = e_known & (pos >= 0)
    pos_c = np.maximum(pos, 0)
    cand = order[pos_c]
    ok &= f_rank[cand] == e_rank_c
    if ttl is not None:
        ok &= (ev_ts - avail[cand]) <= ttl
    return np.where(ok, cand, -1)


def latest_index(ev_entity, feat_entity, feat_ts):
    """The common bug: the latest value ever written for the entity, whatever the event time."""
    ev_entity, feat_entity, feat_ts = np.asarray(ev_entity), np.asarray(feat_entity), np.asarray(feat_ts)
    order = np.lexsort((feat_ts, feat_entity))
    fe = feat_entity[order]
    last = np.r_[fe[1:] != fe[:-1], True]
    ents, idx = fe[last], order[last]
    pos = np.minimum(np.searchsorted(ents, ev_entity), len(ents) - 1)
    return np.where(ents[pos] == ev_entity, idx[pos], -1)


def day_granularity_index(ev_entity, ev_ts, feat_entity, feat_ts, day=86400):
    """Second common bug: joining on calendar dates, so same-day values later than the event are used."""
    return asof_index(ev_entity, (np.asarray(ev_ts) // day) * day + (day - 1), feat_entity, feat_ts)


def take(values, idx, fill=np.nan):
    values = np.asarray(values, dtype=float)
    out = values[np.maximum(idx, 0)]
    return np.where(idx >= 0, out, fill)


def leakage_audit(idx, feat_ts, ev_ts, delay=0):
    """Share of joined rows that used a value written after the event (future leak) or not yet available
    at the event (unavailable). Rows with no match are excluded."""
    m = idx >= 0
    ft = np.asarray(feat_ts, dtype=np.int64)[idx[m]]
    et = np.asarray(ev_ts, dtype=np.int64)[m]
    return {"rows_joined": int(m.sum()), "future_written": float((ft > et).mean()) if m.any() else 0.0,
            "not_yet_available": float((ft + delay > et).mean()) if m.any() else 0.0}


def brute_force_index(ev_entity, ev_ts, feat_entity, feat_ts, *, delay=0, ttl=None, allow_equal=True):
    """Slow reference implementation, used only in tests."""
    out = np.full(len(ev_entity), -1)
    for i, (e, t) in enumerate(zip(ev_entity, ev_ts)):
        best = -1
        for j, (fe, ft) in enumerate(zip(feat_entity, feat_ts)):
            a = ft + delay
            if fe == e and (a <= t if allow_equal else a < t) and (ttl is None or t - a <= ttl):
                if best < 0 or a >= feat_ts[best] + delay:
                    best = j if best < 0 or a > feat_ts[best] + delay else max(best, j)
        out[i] = best
    return out
