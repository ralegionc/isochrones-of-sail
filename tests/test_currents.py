"""Checks for the current climatology added in PLAN.md Addendum 6."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polar as polar_mod  # noqa: E402
from common import haversine_km  # noqa: E402
from currents import Currents, current_at, ground_speed  # noqa: E402
from polar import Polar  # noqa: E402
from router import Graph, replay, route  # noqa: E402
from winds import WindCube, hours  # noqa: E402


def flat_polar(v=6.0):
    return Polar(np.full((len(polar_mod.WS_EDGES) - 1, len(polar_mod.TWA_EDGES) - 1), v))


def test_ground_speed_along_cross_and_head():
    assert np.isclose(ground_speed(6.0, 90.0, 1.0, 0.0, 0.3), 7.0)          # fair current
    assert np.isclose(ground_speed(6.0, 90.0, -1.0, 0.0, 0.3), 5.0)         # foul current
    assert np.isclose(ground_speed(6.0, 90.0, 0.0, 2.0, 0.3), np.sqrt(32))  # cross current
    assert ground_speed(1.0, 90.0, -3.0, 0.0, 0.3) == 0.3                   # swept back: floor


def test_month_interpolation():
    cu = np.arange(12, dtype=float)[:, None, None] * np.ones((12, 3, 3))
    c = Currents(cu, np.zeros_like(cu), np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    mid_march = float(hours(np.datetime64("1841-03-16T12:00")))
    u, _ = current_at(c.params, mid_march, 1.0, 1.0)
    assert abs(u - 2.0) < 0.1  # March is month index 2
    new_year = float(hours(np.datetime64("1841-01-01T00:00")))
    u, _ = current_at(c.params, new_year, 1.0, 1.0)
    assert abs(u - 5.5) < 0.2  # halfway between December (11) and January (0)


def test_replay_with_fair_current():
    cube = WindCube.uniform(0.0, -8.0, days=60)
    th0 = float(hours(np.datetime64("1840-03-01T00:00")))
    lats, lons = [10.0, 10.0], [-30.0, -20.0]
    d_nm = haversine_km(10, -30, 10, -20) / 1.852
    t0 = replay(lats, lons, th0, cube, flat_polar())[-1]
    t1 = replay(lats, lons, th0, cube, flat_polar(), currents=Currents.uniform(1.0, 0.0))[-1]
    assert abs(t0 - d_nm / 6.0) / t0 < 0.01
    assert abs(t1 - d_nm / 7.0) / t1 < 0.01


def test_search_and_replay_agree_with_currents():
    cube = WindCube.uniform(0.0, -8.0, days=80)
    cur = Currents.uniform(-0.8, 0.5)
    th0 = float(hours(np.datetime64("1840-03-01T00:00")))
    toy = 0.5 * (polar_mod.TWA_EDGES[1:] + polar_mod.TWA_EDGES[:-1])
    g = np.interp(toy, [0, 50, 70, 100, 135, 180], [0.15, 0.25, 0.55, 0.85, 1.0, 0.8])
    ws = 0.5 * (polar_mod.WS_EDGES[1:] + polar_mod.WS_EDGES[:-1])
    pol = Polar(np.outer(np.minimum(ws, 12) * 0.55, g))
    res = route(Graph(), 5.0, -30.0, -8.0, -22.0, th0, 24 * 40, polar=pol, cube=cube, currents=cur)
    lats, lons = res["path"]
    t = replay(lats, lons, th0, cube, pol, currents=cur)[-1]
    assert abs(t / res["t_h"] - 1) < 1e-6


def test_segment_table_takes_current_off_the_ground_run():
    cube = WindCube.uniform(0.0, -8.0, days=60)
    t = pd.date_range("1840-03-01 12:00", periods=3, freq="D")
    lon = -30.0 + np.arange(3) * (6.0 * 24 * 1.852 / 111.2 / np.cos(np.radians(10)))  # 6 kn east
    fixes = pd.DataFrame(dict(leg_id="a", time=t, latitude=10.0, longitude=lon, log_wdir=np.nan, log_wspd=np.nan))
    seg = polar_mod.segment_table(fixes, cube, currents=Currents.uniform(1.0, 0.0))
    assert np.allclose(seg.speed_kn, 5.0, atol=0.1)
    assert np.allclose(seg.course, 90.0, atol=1.0)
