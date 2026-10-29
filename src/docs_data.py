"""Export compact data for the interactive page (docs/data/voyages.json, coast.json, hero.json)."""
import gzip
import json

import contourpy
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from common import DERIVED, DOCS, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW, RESULTS

OUT = DOCS / "data"
OUT.mkdir(parents=True, exist_ok=True)


def thin(lat, lon, t=None, tol=0.15):
    """Drop vertices closer than tol degrees to the last kept one (always keep the ends)."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    keep = [0]
    for k in range(1, len(lat) - 1):
        if abs(lat[k] - lat[keep[-1]]) + abs(lon[k] - lon[keep[-1]]) >= tol:
            keep.append(k)
    keep.append(len(lat) - 1)
    out = [np.round(lat[keep], 2).tolist(), np.round(lon[keep], 2).tolist()]
    if t is not None:
        out.append(np.round(np.asarray(t)[keep], 2).tolist())
    return out


def coast():
    g = gpd.read_file(RAW / "naturalearth" / "ne_50m_land.shp")
    g = g.clip(box(LON_MIN - 2, LAT_MIN - 2, LON_MAX + 2, LAT_MAX + 2))
    rings = []
    for geom in g.geometry.simplify(0.12):
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in polys:
            if p.area < 0.3:
                continue
            xy = np.asarray(p.exterior.coords)
            rings.append([np.round(xy[:, 0], 2).tolist(), np.round(xy[:, 1], 2).tolist()])
    (OUT / "coast.json").write_text(json.dumps(rings, separators=(",", ":")))


def voyages(tag="_hull_s42"):
    res = pd.read_csv(RESULTS / f"legs_routed{tag}.csv", parse_dates=["t0", "t1"])
    with gzip.open(RESULTS / f"paths{tag}.json.gz", "rt") as f:
        paths = json.load(f)
    out = []
    for _, r in res.iterrows():
        p = paths[r.leg_id]
        scale = r.t_act_h / r.t_rep_h  # put all three ships on the logged clock
        act = thin(p["act"][0], p["act"][1], np.asarray(p["act"][3]) * scale, tol=0.0)
        opt = thin(p["opt"][0], p["opt"][1], np.asarray(p["opt"][2]) * scale)
        clim = thin(p["clim"][0], p["clim"][1], np.asarray(p["clim"][2]) * scale)
        # stored paths keep every second vertex; finish both ghosts at the leg's end at their true time
        for path, t_end in ((opt, r.t_opt_h), (clim, r.t_clim_h)):
            path[0].append(round(float(act[0][-1]), 2))
            path[1].append(round(float(act[1][-1]), 2))
            path[2].append(round(float(t_end * scale / 24), 2))
        out.append(dict(id=r.leg_id, ship=str(r.ship).title(), frm=str(r.voyage_from).title(),
                        to=str(r.voyage_to).title(), t0=f"{r.t0:%Y-%m-%d}", route=r.route,
                        days=round(r.t_act_h / 24, 1), rh=round(r.R_hind, 3), rc=round(r.R_clim, 3),
                        act=act, opt=opt, clim=clim))
    (OUT / "voyages.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(out)} voyages, {(OUT / 'voyages.json').stat().st_size / 1e6:.1f} MB")


def hero():
    z = np.load(DERIVED / "hero_field.npz")
    field = z["field"][::2, ::2]
    lat, lon = z["lat"][::2], z["lon"][::2]
    gen = contourpy.contour_generator(lon, lat, np.where(np.isfinite(field), field, np.nan))
    levels = []
    for lv in np.arange(2, np.nanmax(field), 2):
        segs = []
        for line in gen.lines(lv):
            if len(line) < 3:
                continue
            segs.append([np.round(line[:, 1], 2).tolist(), np.round(line[:, 0], 2).tolist()])
        if segs:
            levels.append(dict(day=float(lv), lines=segs))
    (OUT / "hero.json").write_text(json.dumps(dict(leg_id=str(z["leg_id"]), levels=levels), separators=(",", ":")))


def main():
    coast()
    voyages()
    hero()
    summary = json.loads((RESULTS / "summary.json").read_text())
    checks = json.loads((RESULTS / "checks.json").read_text())
    slim_checks = {k: {kk: vv for kk, vv in v.items() if kk != "months"}
                   for k, v in checks.items() if k in ("K3", "K4", "K5", "K6")}
    slim_checks["K3_variants"] = {t: dict(name=v["name"], n_ok=v["n_ok"], passed=v["passed"])
                                  for t, v in checks["K3_variants"].items()}
    slim = dict(summary=summary, checks=slim_checks)
    (OUT / "summary.json").write_text(json.dumps(slim, indent=1))


if __name__ == "__main__":
    main()
