import numpy as np
from jikan.join import asof_index, brute_force_index, day_granularity_index, latest_index, leakage_audit, take

FE = np.array([1, 1, 1, 2]); FT = np.array([10, 20, 30, 15])


def test_hand_cases_boundaries_and_unknown_entities():
    ev_e = np.array([1, 1, 1, 1, 2, 3]); ev_t = np.array([9, 10, 25, 99, 14, 50])
    assert list(asof_index(ev_e, ev_t, FE, FT)) == [-1, 0, 1, 2, -1, -1]
    assert list(asof_index(ev_e, ev_t, FE, FT, allow_equal=False)) == [-1, -1, 1, 2, -1, -1]


def test_delay_and_ttl():
    ev_e = np.array([1, 1]); ev_t = np.array([25, 25])
    assert list(asof_index(ev_e, ev_t, FE, FT, delay=0)) == [1, 1]
    assert list(asof_index(ev_e, ev_t, FE, FT, delay=10)) == [0, 0]          # value at 20 only available at 30
    assert list(asof_index([1], [25], FE, FT, ttl=4)) == [-1]                # 20 is 5 old
    assert list(asof_index([1], [25], FE, FT, ttl=5)) == [1]


def test_matches_brute_force_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(30):
        fe = rng.integers(0, 6, 40); ft = rng.integers(0, 100, 40)
        ee = rng.integers(0, 8, 30); et = rng.integers(0, 120, 30)
        kw = dict(delay=int(rng.integers(0, 10)), ttl=[None, 15][int(rng.integers(0, 2))], allow_equal=bool(rng.integers(0, 2)))
        got = asof_index(ee, et, fe, ft, **kw); ref = brute_force_index(ee, et, fe, ft, **kw)
        # equal-time duplicates may resolve to different rows; compare the timestamps they carry
        a = np.where(got >= 0, ft[np.maximum(got, 0)], -1); b = np.where(ref >= 0, ft[np.maximum(ref, 0)], -1)
        assert (got >= 0).tolist() == (ref >= 0).tolist() and (a == b).all()


def test_matches_pandas_merge_asof():
    import pandas as pd
    rng = np.random.default_rng(1)
    fe = rng.integers(0, 50, 2000); ft = rng.integers(0, 10**5, 2000)
    ee = rng.integers(0, 60, 500); et = rng.integers(0, 10**5, 500)
    idx = asof_index(ee, et, fe, ft)
    left = pd.DataFrame({"e": ee, "t": et, "k": np.arange(500)}).sort_values("t")
    right = pd.DataFrame({"e": fe, "t": ft}).sort_values("t").assign(ft=lambda d: d["t"])
    m = pd.merge_asof(left, right, on="t", by="e").sort_values("k")
    mine = np.where(idx >= 0, ft[np.maximum(idx, 0)], -1)
    assert (mine == m["ft"].fillna(-1).astype(int).values).all()


def test_latest_value_bug_uses_the_future_and_the_audit_sees_it():
    ev_e = np.array([1]); ev_t = np.array([12])
    bad = latest_index(ev_e, FE, FT); good = asof_index(ev_e, ev_t, FE, FT)
    assert FT[bad[0]] == 30 and FT[good[0]] == 10
    assert leakage_audit(bad, FT, ev_t)["future_written"] == 1.0
    assert leakage_audit(good, FT, ev_t)["future_written"] == 0.0


def test_day_granularity_join_leaks_same_day_values():
    day = 86400
    idx = day_granularity_index([1], [5 * day + 100], [1, 1], [5 * day + 50000, 4 * day], day=day)
    assert idx[0] == 0                                                         # the value written LATER the same day
    assert leakage_audit(idx, [5 * day + 50000, 4 * day], [5 * day + 100])["future_written"] == 1.0


def test_audit_flags_unavailable_values_when_delay_is_ignored():
    idx = asof_index([1], [25], FE, FT, delay=0)                               # uses ts 20
    assert leakage_audit(idx, FT, [25], delay=10)["not_yet_available"] == 1.0
    assert leakage_audit(idx, FT, [25], delay=3)["not_yet_available"] == 0.0


def test_take_fills_missing():
    out = take([5.0, 6.0], np.array([1, -1, 0]), fill=-9)
    assert list(out) == [6.0, -9.0, 5.0]
