"""Hypothesis tests H1-H3 for every routed model, plus exploratory summaries.

The model whose numbers answer H1-H3 is chosen by the rule in PLAN.md Addendum 3
(written before any voyage was routed) and recorded in results/checks.json.
Writes results/summary.json.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS

N_BOOT = 2000
MODELS = {"_s42": "reanalysis polar, 42S limit (primary by the Addendum 3 rule)",
          "_hull_s42": "tacking-hull polar, 42S limit (exploratory, Addendum 4)",
          "": "reanalysis polar, no limit (as first pre-registered)",
          "_logged_s42": "logged-wind polar, 42S limit (exploratory, Addendum 2)",
          "_cur_s42": "reanalysis polar, currents, 42S limit (exploratory, Addendum 6)",
          "_hull_cur_s42": "tacking-hull polar, currents, 42S limit (exploratory, Addendum 6)"}


def cluster_boot(df, fn, n=N_BOOT, seed=1848):
    """Bootstrap a statistic by resampling ships (all legs of a ship move together)."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    groups = [np.asarray(ix) for ix in df.groupby("ship_key").indices.values()]
    k = len(groups)
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, k, k)])
        vals.append(fn(df.iloc[idx]))
    return np.nanpercentile(vals, [2.5, 97.5]).tolist()


def describe(df, col):
    lo, hi = cluster_boot(df, lambda d: d[col].median())
    return dict(median=float(df[col].median()), ci95=[lo, hi], mean=float(df[col].mean()),
                q25=float(df[col].quantile(0.25)), q75=float(df[col].quantile(0.75)), n=int(len(df)))


def load(tag):
    f = RESULTS / f"legs_routed{tag}.csv"
    if not f.exists():
        return None
    res = pd.read_csv(f, parse_dates=["t0", "t1"])
    res["year"] = res.t0.dt.year + (res.t0.dt.dayofyear - 1) / 365.25
    res["R_hind"] = res.t_rep_h / res.t_opt_h
    res["R_clim"] = res.t_rep_h / res.t_clim_h
    res["R_weather"] = res.t_clim_h / res.t_opt_h
    # days on the logged clock: scale model times by this voyage's logged/replayed ratio
    k = res.t_act_h / res.t_rep_h
    res["days_lost_hind"] = (res.t_rep_h - res.t_opt_h) * k / 24
    res["days_lost_clim"] = (res.t_rep_h - res.t_clim_h) * k / 24
    res.to_csv(f, index=False)
    return res


def analyze(res):
    out = dict(n_legs=int(len(res)), n_ships=int(res.ship_key.nunique()))
    h1 = describe(res, "R_hind")
    h1["in_band_1.10_1.30"] = bool(1.10 <= h1["median"] <= 1.30)
    out["H1"] = h1
    h2 = describe(res, "R_clim")
    lo, hi = h2["ci95"]
    h2["verdict"] = "supported" if hi <= 1.0 else ("rejected" if lo > 1.0 else "inconclusive")
    out["H2"] = h2
    rho = stats.spearmanr(res.year, res.R_clim)
    early, late = res[res.year < 1845], res[res.year >= 1850]

    def dfn(d):
        return d[d.year >= 1850].R_clim.median() - d[d.year < 1845].R_clim.median()

    diff = float(dfn(res))
    out["H3"] = dict(spearman_rho=float(rho.statistic), p=float(rho.pvalue),
                     median_pre1845=float(early.R_clim.median()), n_pre1845=int(len(early)),
                     median_post1850=float(late.R_clim.median()), n_post1850=int(len(late)),
                     diff_post_minus_pre=diff, diff_ci95=cluster_boot(res, dfn),
                     supported=bool(rho.statistic < 0 and rho.pvalue < 0.05 and diff < 0))
    out["R_weather"] = describe(res, "R_weather")
    out["by_route"] = {r: dict(n=int(len(g)), R_hind=float(g.R_hind.median()), R_clim=float(g.R_clim.median()),
                               days_act=float(g.t_act_h.median() / 24),
                               days_lost_hind=float(g.days_lost_hind.median()),
                               days_lost_clim=float(g.days_lost_clim.median()))
                       for r, g in res.groupby("route")}
    ob = res[res.route == "outbound"].dropna(subset=["eq_lon_act", "eq_lon_opt", "eq_lon_clim"])
    out["equator_outbound"] = dict(n=int(len(ob)), act_median=float(ob.eq_lon_act.median()),
                                   opt_median=float(ob.eq_lon_opt.median()),
                                   clim_median=float(ob.eq_lon_clim.median()),
                                   wilcoxon_act_vs_clim_p=float(stats.wilcoxon(ob.eq_lon_act - ob.eq_lon_clim).pvalue),
                                   # exploratory: did ships that crossed further west lose more time?
                                   rho_Rclim_vs_eq_act=float(stats.spearmanr(ob.eq_lon_act, ob.R_clim).statistic),
                                   p_Rclim_vs_eq_act=float(stats.spearmanr(ob.eq_lon_act, ob.R_clim).pvalue))
    out["rep_vs_act"] = dict(median_ratio=float((res.t_rep_h / res.t_act_h).median()),
                             spearman=float(stats.spearmanr(res.t_rep_h, res.t_act_h).statistic))
    return out


def crossing_lon(lats, lons, lat0=30.0):
    """Longitude where a route first crosses latitude lat0, in either direction."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    k = np.where(np.sign(lats[:-1] - lat0) != np.sign(lats[1:] - lat0))[0]
    if not len(k):
        return np.nan
    k = k[0]
    a, b = lats[k], lats[k + 1]
    f = (lat0 - a) / (b - a) if b != a else 0.0
    return float(lons[k] + f * (lons[k + 1] - lons[k]))


def homeward_30n(tag, res):
    """Where homeward routes cross 30N (Addendum 6, test c), from the saved paths."""
    import gzip
    with gzip.open(RESULTS / f"paths{tag}.json.gz", "rt") as f:
        paths = json.load(f)
    rows = []
    for lid in res[res.route == "homeward"].leg_id:
        p = paths[lid]
        rows.append((crossing_lon(*p["act"][:2]), crossing_lon(*p["opt"][:2]), crossing_lon(*p["clim"][:2])))
    d = pd.DataFrame(rows, columns=["act", "opt", "clim"]).dropna()
    return dict(n=int(len(d)), act_median=float(d.act.median()), opt_median=float(d.opt.median()),
                clim_median=float(d.clim.median()))


def lag_cells(tag):
    lag = pd.read_csv(RESULTS / f"lag_segments{tag}.csv")
    lag = lag[(lag.dt_act_h > 12) & (lag.dt_act_h < 36) & np.isfinite(lag.dlag_h)]
    lag["cell_lat"] = np.floor(lag.lat / 5) * 5 + 2.5
    lag["cell_lon"] = np.floor(lag.lon / 5) * 5 + 2.5
    g = lag.groupby(["cell_lat", "cell_lon"])
    cells = pd.DataFrame(dict(n=g.size(), loss_h_per_day=g.dlag_h.sum() / g.dt_rep_h.sum() * 24)).reset_index()
    cells = cells[cells.n >= 30]
    cells.to_csv(RESULTS / f"lag_cells{tag}.csv", index=False)
    return int(len(cells))


def main():
    checks = json.loads((RESULTS / "checks.json").read_text())
    primary = checks["primary"]["tag"]
    gates = {k: checks[k]["passed"] for k in ("K4", "K5", "K6")}
    gates["K3_pre_registered"] = checks["K3"]["passed"]
    gates["K3_by_model"] = {t: v["passed"] for t, v in checks["K3_variants"].items()}
    out = dict(gates=gates, primary=primary, models={})
    for tag, name in MODELS.items():
        res = load(tag)
        if res is None:
            continue
        m = analyze(res)
        m["name"] = name
        m["k3_passed"] = checks["K3_variants"].get(tag or "main", {}).get("passed")
        m["lag_cells"] = lag_cells(tag)
        m["homeward_30N"] = homeward_30n(tag, res)
        out["models"][tag or "none"] = m
        print(f"{name}: n={m['n_legs']} R_hind {m['H1']['median']:.3f} {np.round(m['H1']['ci95'], 3)}  "
              f"R_clim {m['H2']['median']:.3f} {np.round(m['H2']['ci95'], 3)} ({m['H2']['verdict']})  "
              f"H3 rho {m['H3']['spearman_rho']:.3f} p={m['H3']['p']:.3f}  "
              f"eq clim {m['equator_outbound']['clim_median']:.1f}  30N home act/clim "
              f"{m['homeward_30N']['act_median']:.1f}/{m['homeward_30N']['clim_median']:.1f}", flush=True)
    p = out["models"][primary]
    out["n_legs"], out["n_ships"] = p["n_legs"], p["n_ships"]
    out["H1"], out["H2"], out["H3"] = p["H1"], p["H2"], p["H3"]
    out["H1"]["interpretable"] = bool(gates["K4"] and gates["K5"])
    out["flag"] = None if checks["primary"]["k3_passed"] else \
        "The primary model failed its rediscovery check (K3); read H1-H3 with that in mind."
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
