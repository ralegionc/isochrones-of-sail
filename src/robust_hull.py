"""Exploratory (PLAN.md Addendum 4): a polar with a square-rigger's no-go zone.

From 67.5 degrees off the wind outwards the reanalysis-space polar is kept. Closer
than that, the only way to make ground is to tack at 67.5 degrees, which makes good
v(67.5) * cos(67.5) / cos(angle) along the wanted direction.
"""
import json

import numpy as np
import pandas as pd

import climatology
import polar as polar_mod
from common import DERIVED, RESULTS, SOUTH_LIMIT
from polar import TWA_EDGES, Polar
from prepare import calibrate, k3_rediscovery
from router import Graph
from winds import WindCube

NO_GO = 67.5


def hull_table(tab):
    centres = 0.5 * (TWA_EDGES[1:] + TWA_EDGES[:-1])
    base = Polar(tab)
    out = tab.copy()
    ws = 0.5 * (polar_mod.WS_EDGES[1:] + polar_mod.WS_EDGES[:-1])
    for a, w in enumerate(ws):
        v_ng = base(w, NO_GO)[0]
        for b, c in enumerate(centres):
            if c < NO_GO:
                out[a, b] = v_ng * np.cos(np.radians(NO_GO)) / np.cos(np.radians(c))
    return out


def main():
    d = json.loads((DERIVED / "polar.json").read_text())
    tab = hull_table(np.array(d["table"]) / d.get("calibration", 1.0))
    legs = pd.read_csv(DERIVED / "legs_clean.csv", parse_dates=["t0", "t1"])
    fixes = pd.read_csv(DERIVED / "fixes.csv", parse_dates=["time"])
    fixes = fixes[fixes.leg_id.isin(legs.leg_id)]
    cube = WindCube.load_all()
    scale = calibrate(Polar(tab), legs, fixes, cube)
    tab = tab * scale
    polar_mod.save(tab, np.array(d["count"]), extra=dict(calibration=scale, no_go_deg=NO_GO),
                   path=DERIVED / "polar_hull.json")
    pol = Polar(tab)
    clim = climatology.build(cube, pol, path=DERIVED / "clim_speed_hull.npz")
    r = k3_rediscovery(Graph(lat_limit=SOUTH_LIMIT), pol, clim)
    checks = json.loads((RESULTS / "checks.json").read_text())
    checks.setdefault("K3_variants", {})["_hull_s42"] = dict(name="tacking-hull polar, 42S limit", **r)
    (RESULTS / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    print(f"hull polar: calibration {scale:.3f}; K3 {r['n_ok']}/12",
          [(m["month"], round(m["equator_lon"], 1), round(m["lat_at_60E"], 1)) for m in r["months"]])


if __name__ == "__main__":
    main()
