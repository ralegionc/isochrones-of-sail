"""Clean the legs, fit the polar, build the climatology, run known-answer checks K3-K6.

Writes data/derived/legs_clean.csv, data/derived/segments.csv, data/derived/polar.json,
data/derived/clim_speed.npz and results/checks.json.
"""
import json
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import climatology
import polar as polar_mod
from common import DERIVED, RESULTS, YEAR_MAX, YEAR_MIN, angle_diff, haversine_km
from polar import Polar
from router import Graph, replay, route
from voyages import load_legs
from winds import WindCube, hours

SHIFTS = [-2, -1, 1, 2]


def clean_legs(legs, fixes):
    legs = legs.copy()
    legs["gc_km"] = haversine_km(legs.lat0, legs.lon0, legs.lat1, legs.lon1)
    track = fixes.groupby("leg_id").apply(
        lambda g: haversine_km(g.latitude.values[:-1], g.longitude.values[:-1],
                               g.latitude.values[1:], g.longitude.values[1:]).sum(), include_groups=False)
    legs["track_km"] = legs.leg_id.map(track)
    legs["sinuosity"] = legs.track_km / legs.gc_km
    n0 = len(legs)
    legs = legs[(legs.gc_km >= 1500) & (legs.sinuosity <= 3.0)]
    n1 = len(legs)
    # duplicate logbooks of the same voyage (e.g. captain's and mate's copies)
    legs = legs.assign(ship_key=legs.ship.fillna("?").str.upper().str.strip())
    legs = legs.sort_values("n_fix", ascending=False)
    kept = []
    for _, r in legs.iterrows():
        dup = False
        for k in kept:
            if k.ship_key != r.ship_key:
                continue
            ov = (min(k.t1, r.t1) - max(k.t0, r.t0)).total_seconds()
            if ov > 0.5 * min((k.t1 - k.t0).total_seconds(), (r.t1 - r.t0).total_seconds()):
                dup = True
                break
        if not dup:
            kept.append(r)
    legs = pd.DataFrame(kept).sort_values("t0")
    # winds exist 1836-01-01 to 1855-12-31; keep legs whose search horizon fits
    legs = legs[(legs.t0 >= "1836-01-03") & (legs.t1 + pd.Timedelta(days=60) <= "1855-12-31")]
    print(f"legs: {n0} -> {n1} after distance/sinuosity -> {len(legs)} after dedup and wind coverage")
    return legs, dict(n_raw=n0, n_dist=n1, n_clean=len(legs))


def shifted_segments(fixes, cube, shift):
    seg = polar_mod.segment_table(fixes, cube, year_shift=shift)
    if shift:
        ts = pd.DatetimeIndex(seg.t_a) + pd.DateOffset(years=shift)
        ok = (ts >= f"{YEAR_MIN}-01-02") & (ts <= f"{YEAR_MAX}-12-29")
        seg = seg[ok]
    return seg


def fold_of(ship):
    return zlib.crc32(str(ship).encode()) % 2


def k4_wind_direction(real, placebo):
    r = real.dropna(subset=["log_wdir"]).copy()
    r["err"] = angle_diff(r.log_wdir, r.wind_from)
    pl = []
    for s, seg in placebo.items():
        x = seg.dropna(subset=["log_wdir"])
        pl.append(pd.Series(np.asarray(angle_diff(x.log_wdir, x.wind_from)),
                            index=list(zip(x.leg_id, x.t_a)), name=s))
    pl = pd.concat(pl, axis=1).mean(axis=1)
    r["err_placebo"] = pd.Series(list(zip(r.leg_id, r.t_a))).map(pl).values
    r = r.dropna(subset=["err_placebo"])
    per_leg = r.groupby("leg_id")[["err", "err_placebo"]].mean()
    diff = per_leg.err - per_leg.err_placebo
    w = stats.wilcoxon(diff, alternative="less")
    # systematic offset between logged and reanalysis direction (compass variation etc.)
    off = ((r.log_wdir - r.wind_from + 180) % 360 - 180)
    return dict(n_obs=int(len(r)), n_legs=int(len(per_leg)),
                median_err_real=float(r.err.median()), median_err_placebo=float(r.err_placebo.median()),
                median_leg_diff=float(diff.median()), frac_legs_real_better=float((diff < 0).mean()),
                wilcoxon_p=float(w.pvalue), circ_mean_offset_deg=float(np.degrees(np.angle(
                    np.exp(1j * np.radians(off)).mean()))),
                passed=bool(w.pvalue < 0.01 and diff.median() < 0))


def predicted_speed(seg, pol):
    return pol(seg.tws.values, seg.twa.values)


def k5_daily_progress(real, placebo, legs):
    fold = legs.set_index("leg_id").ship_key.map(fold_of)
    out = {}

    def cross_pred(seg):
        f = seg.leg_id.map(fold).values
        pred = np.full(len(seg), np.nan)
        for k in (0, 1):
            tab, _ = polar_mod.fit(seg[f != k])
            pred[f == k] = Polar(tab)(seg.tws.values[f == k], seg.twa.values[f == k])
        return pred

    def per_leg_rho(seg, pred):
        d = seg.assign(pred=pred)
        res = {}
        for lid, g in d.groupby("leg_id"):
            if len(g) >= 10 and g.pred.std() > 0:
                res[lid] = stats.spearmanr(g.speed_kn, g.pred).statistic
        return pd.Series(res)

    rho_real = per_leg_rho(real, cross_pred(real))
    rhos = [per_leg_rho(seg, cross_pred(seg)) for seg in placebo.values()]
    rho_pl = pd.concat(rhos, axis=1).mean(axis=1)
    both = pd.concat([rho_real.rename("real"), rho_pl.rename("placebo")], axis=1).dropna()
    diff = both.real - both.placebo
    w = stats.wilcoxon(diff, alternative="greater")
    pooled_real = stats.spearmanr(real.speed_kn, cross_pred(real)).statistic
    out = dict(n_legs=int(len(both)), median_rho_real=float(both.real.median()),
               median_rho_placebo=float(both.placebo.median()), median_leg_diff=float(diff.median()),
               frac_legs_real_better=float((diff > 0).mean()), wilcoxon_p=float(w.pvalue),
               pooled_rho_real=float(pooled_real),
               passed=bool(w.pvalue < 0.01 and diff.median() > 0 and both.real.median() > 0))
    return out


def k6_polar(pol):
    close = float(np.mean(pol(np.full(4, 8.0), [7.5, 22.5, 37.5, 52.5])))
    broad = float(np.mean(pol(np.full(3, 8.0), [120.0, 135.0, 150.0])))
    p8 = float(pol(8.0, 135.0)[0])
    p3 = float(pol(3.0, 135.0)[0])
    return dict(close_hauled_8ms=close, broad_reach_8ms=broad, reach_8ms=p8, reach_3ms=p3,
                passed=bool(close < broad and p8 > p3))


def k3_rediscovery(graph, pol, clim, currents=None):
    start, target = (48.5, -8.0), (-7.0, 104.0)
    rows = []
    for m in range(1, 13):
        th0 = float(hours(np.datetime64(f"1845-{m:02d}-01T00:00")))
        res = route(graph, *start, *target, th0, 24 * 220, polar=pol, clim=clim, currents=currents)
        lats, lons = res["path"]
        k = np.where((lats[:-1] > 0) & (lats[1:] <= 0))[0]
        eq_lon = float(np.interp(0, [lats[k[0] + 1], lats[k[0]]], [lons[k[0] + 1], lons[k[0]]])) if len(k) else np.nan
        k60 = np.where(lons >= 60)[0]
        lat60 = float(lats[k60[0]]) if len(k60) else np.nan
        ok = (-35 <= eq_lon <= -15) and (lat60 < -30)
        rows.append(dict(month=m, days=float(res["t_h"] / 24), equator_lon=eq_lon, lat_at_60E=lat60,
                         ok=bool(ok), path=[lats[::4].round(2).tolist(), lons[::4].round(2).tolist()]))
    n_ok = sum(r["ok"] for r in rows)
    return dict(months=rows, n_ok=n_ok, passed=bool(n_ok >= 10))


def calibrate(pol, legs, fixes, cube, currents=None):
    """Global speed factor so the median replayed leg takes its logged time (Addendum 1)."""
    ratios = []
    for lid, g in fixes.groupby("leg_id"):
        th0 = float(hours(g.time.values[0]))
        t_act = (g.time.values[-1] - g.time.values[0]) / np.timedelta64(1, "h")
        rep = replay(g.latitude.values, g.longitude.values, th0, cube, pol, currents=currents)[-1]
        ratios.append(rep / t_act)
    return float(np.median(ratios))


def main():
    legs, fixes = load_legs()
    legs, counts = clean_legs(legs, fixes)
    fixes = fixes[fixes.leg_id.isin(legs.leg_id)]
    legs.to_csv(DERIVED / "legs_clean.csv", index=False)
    print("loading winds", flush=True)
    cube = WindCube.load_all()
    real = shifted_segments(fixes, cube, 0)
    real.to_csv(DERIVED / "segments.csv", index=False)
    placebo = {s: shifted_segments(fixes, cube, s) for s in SHIFTS}
    print(f"{len(real)} ship-days with real winds", flush=True)

    tab, cnt = polar_mod.fit(real)
    scale = calibrate(Polar(tab), legs, fixes, cube)
    print(f"calibration factor {scale:.3f}", flush=True)
    tab = tab * scale
    polar_mod.save(tab, cnt, extra=dict(n_days=int(len(real)), calibration=scale))
    pol = Polar(tab)
    checks = dict(counts=counts)
    checks["K4"] = k4_wind_direction(real, placebo)
    print("K4", checks["K4"], flush=True)
    checks["K5"] = k5_daily_progress(real, placebo, legs)
    print("K5", checks["K5"], flush=True)
    checks["K6"] = k6_polar(pol)
    print("K6", checks["K6"], flush=True)

    print("climatology", flush=True)
    clim = climatology.build(cube, pol)
    graph = Graph()
    checks["K3"] = k3_rediscovery(graph, pol, clim)
    print("K3", {k: v for k, v in checks["K3"].items() if k != "months"},
          [(r["month"], round(r["equator_lon"], 1), round(r["lat_at_60E"], 1)) for r in checks["K3"]["months"]],
          flush=True)
    (RESULTS / "checks.json").write_text(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    main()
