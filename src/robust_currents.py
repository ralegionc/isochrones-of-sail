"""Exploratory (PLAN.md Addendum 6): refit the polars on speed through the water and rerun K3.

Builds, for the reanalysis polar and the tacking-hull polar:
  data/derived/polar{,_hull}_cur.json        polar fitted on ground run minus current, recalibrated
  data/derived/clim_speed{,_hull}_cur.npz    monthly expected boat speed per heading
and adds K3 results "_cur_s42" and "_hull_cur_s42" (42S limit, currents on) to results/checks.json.
"""
import json

import numpy as np
import pandas as pd

import climatology
import polar as polar_mod
from common import DERIVED, RESULTS, SOUTH_LIMIT
from currents import Currents
from polar import Polar
from prepare import calibrate, k3_rediscovery
from robust_hull import hull_table
from router import Graph
from winds import WindCube


def main():
    legs = pd.read_csv(DERIVED / "legs_clean.csv", parse_dates=["t0", "t1"])
    fixes = pd.read_csv(DERIVED / "fixes.csv", parse_dates=["time"])
    fixes = fixes[fixes.leg_id.isin(legs.leg_id)]
    cube = WindCube.load_all()
    cur = Currents.load()
    seg = polar_mod.segment_table(fixes, cube, currents=cur)
    seg.to_csv(DERIVED / "segments_cur.csv", index=False)
    raw, cnt = polar_mod.fit(seg)
    checks = json.loads((RESULTS / "checks.json").read_text())
    graph = Graph(lat_limit=SOUTH_LIMIT)
    for variant, tab, name in (("_cur", raw, "reanalysis polar, currents, 42S limit"),
                               ("_hull_cur", hull_table(raw), "tacking-hull polar, currents, 42S limit")):
        scale = calibrate(Polar(tab), legs, fixes, cube, currents=cur)
        tab = tab * scale
        polar_mod.save(tab, cnt, extra=dict(n_days=int(len(seg)), calibration=scale, currents=True),
                       path=DERIVED / f"polar{variant}.json")
        pol = Polar(tab)
        clim = climatology.build(cube, pol, path=DERIVED / f"clim_speed{variant}.npz")
        r = k3_rediscovery(graph, pol, clim, currents=cur)
        checks.setdefault("K3_variants", {})[f"{variant}_s42"] = dict(name=name, **r)
        close = float(np.mean(pol(np.full(4, 8.0), [7.5, 22.5, 37.5, 52.5])))
        broad = float(np.mean(pol(np.full(3, 8.0), [120.0, 135.0, 150.0])))
        print(f"{name}: calibration {scale:.3f}, close-hauled {close:.2f} kn vs broad reach {broad:.2f} kn; "
              f"K3 {r['n_ok']}/12", [(m["month"], round(m["equator_lon"], 1), round(m["lat_at_60E"], 1))
                                    for m in r["months"]], flush=True)
    (RESULTS / "checks.json").write_text(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    main()
