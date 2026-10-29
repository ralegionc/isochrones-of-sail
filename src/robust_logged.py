"""Exploratory robustness (PLAN.md Addendum 2): a polar fitted to the sailors' own logged winds.

The main polar is fitted against reanalysis winds, whose errors blur the angle
dependence (headwinds look less costly than they were). Here speed made good is
binned by the logged wind force and the logged direction instead, after removing
the fleet-wide mean offset between logged and reanalysis directions (compass
variation was often left uncorrected). Routing still uses the reanalysis winds.
"""
import json

import numpy as np
import pandas as pd

import climatology
import polar as polar_mod
from common import DERIVED, RESULTS, angle_diff
from polar import Polar
from prepare import calibrate
from winds import WindCube


def main():
    checks = json.loads((RESULTS / "checks.json").read_text())
    off = checks["K4"]["circ_mean_offset_deg"]
    legs = pd.read_csv(DERIVED / "legs_clean.csv", parse_dates=["t0", "t1"])
    fixes = pd.read_csv(DERIVED / "fixes.csv", parse_dates=["time"])
    fixes = fixes[fixes.leg_id.isin(legs.leg_id)]
    seg = pd.read_csv(DERIVED / "segments.csv").dropna(subset=["log_wdir", "log_wspd"])
    seg = seg.assign(tws=seg.log_wspd, twa=angle_diff(seg.course, (seg.log_wdir - off) % 360))
    tab, cnt = polar_mod.fit(seg)
    cube = WindCube.load_all()
    scale = calibrate(Polar(tab), legs, fixes, cube)
    tab = tab * scale
    polar_mod.save(tab, cnt, extra=dict(n_days=int(len(seg)), calibration=scale, compass_offset=off),
                   path=DERIVED / "polar_logged.json")
    pol = Polar(tab)
    climatology.build(cube, pol, path=DERIVED / "clim_speed_logged.npz")
    close = float(np.mean(pol(np.full(4, 8.0), [7.5, 22.5, 37.5, 52.5])))
    broad = float(np.mean(pol(np.full(3, 8.0), [120.0, 135.0, 150.0])))
    print(f"logged-wind polar: offset {off:.1f} deg, calibration {scale:.3f}, "
          f"close-hauled {close:.2f} kn vs broad reach {broad:.2f} kn at 8 m/s")


if __name__ == "__main__":
    main()
