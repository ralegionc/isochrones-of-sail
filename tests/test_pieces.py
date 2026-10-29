"""Unit checks for geodesy, voyage cleaning, the polar fit, climatology timing and replay."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import angle_diff, bearing_deg, haversine_km, wind_from_uv  # noqa: E402
import polar as polar_mod  # noqa: E402
from polar import Polar  # noqa: E402
from climatology import Clim  # noqa: E402
from router import replay  # noqa: E402
from voyages import drop_outliers, split_legs  # noqa: E402
from winds import WindCube, hours  # noqa: E402


def test_haversine_and_bearing():
    assert abs(haversine_km(51.5074, -0.1278, 48.8566, 2.3522) - 343.5) < 2
    assert abs(bearing_deg(0, 0, 0, 10) - 90) < 1e-6
    assert abs(bearing_deg(0, 0, -10, 0) - 180) < 1e-6


def test_wind_direction_convention():
    frm, spd = wind_from_uv(0.0, -5.0)  # blowing towards the south = northerly wind
    assert abs(frm - 0) < 1e-9 or abs(frm - 360) < 1e-9
    assert spd == 5.0
    frm, _ = wind_from_uv(5.0, 0.0)  # blowing towards the east = westerly wind
    assert abs(frm - 270) < 1e-9
    assert angle_diff(350, 10) == 20


def _fixes(lats, lons, start="1840-05-01 12:00", anchored=None, gaps=None):
    t = pd.date_range(start, periods=len(lats), freq="D")
    if gaps:
        t = t + pd.to_timedelta(np.cumsum([gaps.get(i, 0) for i in range(len(lats))]), unit="D")
    return pd.DataFrame(dict(time=t, latitude=lats, longitude=lons,
                             Anchored=anchored if anchored is not None else np.zeros(len(lats))))


def test_drop_outliers_removes_single_bad_fix():
    lats = np.linspace(40, 30, 11)
    lons = np.full(11, -20.0)
    lats[5] = 10.0  # a 1,200 nm jump and back
    g = drop_outliers(_fixes(lats, lons))
    assert len(g) == 10 and 10.0 not in g.latitude.values


def test_split_on_anchor_and_gap():
    n = 20
    anch = np.zeros(n)
    anch[8] = 1
    g = _fixes(np.linspace(40, 20, n), np.full(n, -20.0), anchored=anch, gaps={14: 5})
    legs = list(split_legs(g))
    assert [len(x) for x in legs] == [8, 5, 6]


def test_polar_fit_recovers_known_speeds():
    rng = np.random.default_rng(0)
    n = 20000
    tws = rng.uniform(0, 14, n)
    twa = rng.uniform(0, 180, n)
    true = 0.5 * np.minimum(tws, 12) * (0.3 + 0.7 * twa / 180)
    seg = pd.DataFrame(dict(tws=tws, twa=twa, speed_kn=true))
    tab, cnt = polar_mod.fit(seg)
    p = Polar(tab)
    for s, a in ((5, 120), (9, 60), (3, 170)):
        assert abs(p(s, a)[0] - 0.5 * s * (0.3 + 0.7 * a / 180)) < 0.35


def test_month_fraction_is_zero_mid_january():
    c = Clim(np.ones((12, 2, 2, 36), np.float32), np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    th = float(hours(np.datetime64("1840-01-16T12:00")))
    mf = c.params_for(th, 48)[1]
    assert abs(mf[0]) < 0.02


def test_replay_straight_line_in_uniform_wind():
    tab = np.full((len(polar_mod.WS_EDGES) - 1, len(polar_mod.TWA_EDGES) - 1), 6.0)
    p = Polar(tab)
    cube = WindCube.uniform(5.0, 0.0, days=60)
    th0 = float(hours(np.datetime64("1840-03-01T00:00")))
    t = replay([10.0, 10.0], [-30.0, -20.0], th0, cube, p)[-1]
    d_nm = haversine_km(10, -30, 10, -20) / 1.852
    assert abs(t - d_nm / 6.0) / t < 0.01
