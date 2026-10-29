"""Route every clean leg three ways and replay each route through the real winds.

T_rep   logged track replayed through 20CR winds with the fleet polar
T_opt   hindsight optimum through the real winds (path found by search, then replayed)
T_clim  optimum through the monthly climatology, then sailed through the real winds

Also records, at every logged fix, how far the ship was behind the hindsight
isochrone frontier (lag), so the loss can be mapped.
"""
import gzip
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from climatology import Clim
from common import DERIVED, RESULTS, SOUTH_LIMIT
from polar import Polar
from router import NX, Graph, replay, route
from winds import WindCube, hours


def equator_lon(lats, lons):
    k = np.where((lats[:-1] > 0) & (lats[1:] <= 0))[0]
    if not len(k):
        return np.nan
    k = k[0]
    return float(np.interp(0, [lats[k + 1], lats[k]], [lons[k + 1], lons[k]]))


def crossing_lon(lats, lons, lat0=30.0):
    """Longitude where a route first crosses latitude lat0, in either direction."""
    lats, lons = np.asarray(lats), np.asarray(lons)
    k = np.where(np.sign(lats[:-1] - lat0) != np.sign(lats[1:] - lat0))[0]
    if not len(k):
        return np.nan
    k = k[0]
    a, b = lats[k], lats[k + 1]
    f = (lat0 - a) / (b - a) if b != a else 0.0
    return float(lons[k] + f * (lons[k + 1] - lons[k]))


def route_leg(leg, fx, graph, cube, pol, clim, currents=None):
    lats, lons = fx.latitude.values, fx.longitude.values
    th0 = float(hours(np.datetime64(leg.t0)))
    t_act = (fx.time.values[-1] - fx.time.values[0]) / np.timedelta64(1, "h")
    t_fix = (fx.time.values - fx.time.values[0]) / np.timedelta64(1, "h")
    rep = replay(lats, lons, th0, cube, pol, currents=currents)
    # the full field is only needed out to where the ship actually got; widen if the target is missed
    t_max = max(1.35 * t_act, t_act + 10 * 24)
    for _ in range(3):
        opt = route(graph, lats[0], lons[0], lats[-1], lons[-1], th0, t_max, polar=pol, cube=cube, full=True,
                    currents=currents)
        if opt["path"] is not None:
            break
        t_max *= 1.6
    if opt["path"] is None:
        return None
    olat, olon = opt["path"]
    o_arr = replay(olat, olon, th0, cube, pol, currents=currents)
    t_opt = o_arr[-1]

    cl = route(graph, lats[0], lons[0], lats[-1], lons[-1], th0, 3.0 * t_act, polar=pol, clim=clim,
               currents=currents)
    if cl["path"] is None:
        return None
    clat, clon = cl["path"]
    c_arr = replay(clat, clon, th0, cube, pol, currents=currents)
    t_clim = c_arr[-1]
    # the hindsight optimum is by definition no slower than any route we know of (Addendum 5)
    if t_clim < t_opt:
        olat, olon, o_arr, t_opt = clat, clon, c_arr, t_clim

    # lag behind the hindsight frontier at each logged fix
    field = opt["field"]
    best = np.array([field[graph.snap(a, b)] for a, b in zip(lats, lons)])
    lag = rep - best

    row = dict(leg_id=leg.leg_id, t_act_h=t_act, t_rep_h=rep[-1], t_opt_h=t_opt, t_clim_h=t_clim,
               t_opt_search_h=opt["t_h"], t_clim_search_h=cl["t_h"],
               eq_lon_act=equator_lon(lats, lons), eq_lon_opt=equator_lon(olat, olon),
               eq_lon_clim=equator_lon(clat, clon),
               lon30_act=crossing_lon(lats, lons), lon30_opt=crossing_lon(olat, olon),
               lon30_clim=crossing_lon(clat, clon),
               n_fix_routed=len(fx))
    seg = pd.DataFrame(dict(leg_id=leg.leg_id, lat=0.5 * (lats[1:] + lats[:-1]),
                            lon=0.5 * (lons[1:] + lons[:-1]), dlag_h=np.diff(lag),
                            dt_act_h=np.diff(t_fix), dt_rep_h=np.diff(rep)))
    paths = dict(act=[lats.round(2).tolist(), lons.round(2).tolist(), (t_fix / 24).round(2).tolist(),
                      (rep / 24).round(2).tolist()],
                 opt=[olat[::2].round(2).tolist(), olon[::2].round(2).tolist(),
                      (o_arr[::2] / 24).round(2).tolist()],
                 clim=[clat[::2].round(2).tolist(), clon[::2].round(2).tolist(),
                       (c_arr[::2] / 24).round(2).tolist()])
    return row, seg, paths


def main(limit=None, workers=15, variant="", south=False, with_currents=False):
    """variant "" uses the 20CR-space polar; "_logged" the polar fitted to logged winds.
    south=True applies the 42S southern limit (PLAN.md Addendum 3); outputs get "_s42".
    with_currents=True adds the drifter current climatology (Addendum 6); outputs get "_cur"."""
    from currents import Currents
    legs = pd.read_csv(DERIVED / "legs_clean.csv", parse_dates=["t0", "t1"])
    fixes = pd.read_csv(DERIVED / "fixes.csv", parse_dates=["time"])
    fixes = fixes[fixes.leg_id.isin(legs.leg_id)]
    groups = {k: g for k, g in fixes.groupby("leg_id")}
    if limit:
        legs = legs.head(limit)
    print(f"routing {len(legs)} legs", flush=True)
    cube = WindCube.load_all()
    cv = variant + ("_cur" if with_currents else "")
    pol = Polar.load(DERIVED / f"polar{cv}.json")
    clim = Clim.load(DERIVED / f"clim_speed{cv}.npz")
    currents = Currents.load() if with_currents else None
    graph = Graph(lat_limit=SOUTH_LIMIT if south else None)
    tag = cv + ("_s42" if south else "")
    rows, segs, paths = [], [], {}

    def job(leg):
        try:
            return route_leg(leg, groups[leg.leg_id], graph, cube, pol, clim, currents)
        except Exception as e:  # keep going; report at the end
            print(f"  {leg.leg_id} failed: {e}", flush=True)
            return None

    with ThreadPoolExecutor(workers) as ex:
        for n, out in enumerate(ex.map(job, [r for _, r in legs.iterrows()])):
            if out is None:
                continue
            row, seg, p = out
            rows.append(row)
            segs.append(seg)
            paths[row["leg_id"]] = p
            if (n + 1) % 50 == 0:
                print(f"  {n + 1}/{len(legs)}", flush=True)
    res = pd.DataFrame(rows).merge(legs, on="leg_id")
    res.to_csv(RESULTS / f"legs_routed{tag}.csv", index=False)
    pd.concat(segs).to_csv(RESULTS / f"lag_segments{tag}.csv", index=False)
    with gzip.open(RESULTS / f"paths{tag}.json.gz", "wt") as f:
        json.dump(paths, f)
    print(f"routed {len(res)} legs", flush=True)


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    variant = "_logged" if "--logged" in sys.argv else ("_hull" if "--hull" in sys.argv else "")
    main(limit=int(args[0]) if args else None, variant=variant, south="--south" in sys.argv,
         with_currents="--currents" in sys.argv)
