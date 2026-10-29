"""Known-answer checks K1 and K2: the router in a uniform wind."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import bearing_deg, haversine_km  # noqa: E402
from polar import TWA_EDGES, WS_EDGES, Polar, convex_speed  # noqa: E402
from router import Graph, replay, route  # noqa: E402
from winds import WindCube, hours  # noqa: E402

TWS = 8.0


def toy_polar():
    """Square-rigger-like shape: poor close-hauled, best on a broad reach."""
    ws = 0.5 * (WS_EDGES[1:] + WS_EDGES[:-1])
    twa = 0.5 * (TWA_EDGES[1:] + TWA_EDGES[:-1])
    g = np.interp(twa, [0, 50, 70, 100, 135, 180], [0.15, 0.25, 0.55, 0.85, 1.0, 0.8])
    f = np.minimum(ws, 12.0) * 0.55
    return Polar(np.outer(f, g))


@pytest.fixture(scope="module")
def setup():
    graph = Graph()
    polar = toy_polar()
    # wind blowing from the north (towards the south) at TWS m/s
    cube = WindCube.uniform(0.0, -TWS)
    th0 = float(hours(np.datetime64("1840-03-01T00:00")))
    return graph, polar, cube, th0


def test_polar_lookup_matches_table():
    p = toy_polar()
    ws = 0.5 * (WS_EDGES[1:] + WS_EDGES[:-1])
    assert np.isclose(p(ws[3], 52.5)[0], p.table[3, 4])


def test_k1_isochrones_match_convex_polar(setup):
    graph, polar, cube, th0 = setup
    lat0, lon0 = 0.0, -30.0
    res = route(graph, lat0, lon0, 10.0, -30.0, th0, 24 * 40, polar=polar, cube=cube, full=True)
    field = res["field"]
    ang, best = convex_speed(polar, TWS)
    errs = []
    for brg in np.arange(0, 360, 15):
        # a point ~900 km away along bearing brg
        d_deg = 900 / 111.2
        lat = lat0 + d_deg * np.cos(np.radians(brg))
        lon = lon0 + d_deg * np.sin(np.radians(brg)) / np.cos(np.radians(lat0))
        i, j = graph.snap(lat, lon)
        la, lo = -62 + 0.25 * i, -80 + 0.25 * j
        dist_nm = haversine_km(lat0, lon0, la, lo) / 1.852
        b = bearing_deg(lat0, lon0, la, lo)
        rel = b  # wind source is north, so bearing is the angle off the wind
        v = np.interp(rel, ang, best, period=360)
        expected = dist_nm / v
        errs.append(field[i, j] / expected - 1)
    errs = np.array(errs)
    # grid paths can only be slower than the continuous optimum, never faster
    assert errs.min() > -0.01, errs
    assert np.abs(errs).max() < 0.03, errs


@pytest.mark.parametrize("target_lat", [12.0, -12.0])
def test_k2_upwind_downwind_times(setup, target_lat):
    graph, polar, cube, th0 = setup
    lat0, lon0 = 0.0, -30.0
    res = route(graph, lat0, lon0, target_lat, lon0, th0, 24 * 60, polar=polar, cube=cube)
    ang, best = convex_speed(polar, TWS)
    v = best[0] if target_lat > 0 else best[len(ang) // 2]
    dist_nm = haversine_km(lat0, lon0, target_lat, lon0) / 1.852
    assert abs(res["t_h"] / (dist_nm / v) - 1) < 0.03
    # replaying the found path gives the same time as the search
    lats, lons = res["path"]
    t = replay(lats, lons, th0, cube, polar)[-1]
    assert abs(t / res["t_h"] - 1) < 0.03
