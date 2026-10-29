"""Monthly climatology of expected speed made good per heading, from 1836-1855 winds.

For each wind grid cell, calendar month and heading (10 degree steps), average the
polar speed over every 6-hourly wind in that month across all years. This is what a
planner with perfect wind charts but no forecasts would know.
"""
import numba as nb
import numpy as np
import pandas as pd

from common import DERIVED
from polar import Polar, polar_speed

CLIM_PATH = DERIVED / "clim_speed.npz"
NH = 36


@nb.njit(parallel=True, cache=True, fastmath=True)
def _accumulate(u, v, month_idx, table, ws, twa, nh):
    nt, ny, nx = u.shape
    out = np.zeros((12, ny, nx, nh), np.float64)
    cnt = np.zeros(12, np.int64)
    for t in range(nt):
        cnt[month_idx[t]] += 1
    for cell in nb.prange(ny * nx):
        i = cell // nx
        j = cell % nx
        for t in range(nt):
            m = month_idx[t]
            uu = u[t, i, j]
            vv = v[t, i, j]
            spd = np.sqrt(uu * uu + vv * vv)
            wfrom = (np.degrees(np.arctan2(-uu, -vv)) + 360.0) % 360.0
            for h in range(nh):
                hdg = h * 360.0 / nh
                a = abs((hdg - wfrom + 180.0) % 360.0 - 180.0)
                out[m, i, j, h] += polar_speed(table, ws, twa, spd, a)
    for m in range(12):
        out[m] /= max(cnt[m], 1)
    return out


def build(cube, polar, path=CLIM_PATH):
    month_idx = (pd.DatetimeIndex(cube.times).month.values - 1).astype(np.int64)
    tab = _accumulate(cube.u, cube.v, month_idx, *polar.params, NH).astype(np.float32)
    np.savez_compressed(path, speed=tab, lat=cube.lat, lon=cube.lon)
    return Clim(tab, cube.lat, cube.lon)


class Clim:
    def __init__(self, tab, lat, lon):
        self.tab = np.ascontiguousarray(tab, dtype=np.float32)
        self.lat0 = float(lat[0])
        self.dlat = float((lat[-1] - lat[0]) / (len(lat) - 1))
        self.lon0 = float(lon[0])
        self.dlon = float((lon[-1] - lon[0]) / (len(lon) - 1))

    @classmethod
    def load(cls, path=CLIM_PATH):
        z = np.load(path)
        return cls(z["speed"], z["lat"], z["lon"])

    def params_for(self, th_start, t_max_h, dt=6.0):
        """Fractional month index (0 = mid January) every dt hours from th_start."""
        t0 = np.datetime64("1800-01-01T00:00") + np.timedelta64(int(round(th_start * 60)), "m")
        times = pd.DatetimeIndex(t0 + np.arange(0, t_max_h + 48, dt).astype("timedelta64[h]"))
        dim = times.days_in_month.values
        frac = (times.day.values - 1 + times.hour.values / 24.0) / dim
        mf = (times.month.values - 1) + frac - 0.5
        return (self.tab, mf.astype(np.float64), float(dt), self.lat0, self.dlat, self.lon0, self.dlon)
