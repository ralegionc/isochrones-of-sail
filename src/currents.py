"""Monthly near-surface current climatology (NOAA Global Drifter Program) for the router.

PLAN.md Addendum 6 (exploratory). Source: Laurindo, Mariano and Lumpkin (2017), drifter
monthly means v3.10 from AOML's ERDDAP, 0.25 degree. Currents are stored in knots; gaps
(land, or no drifters) count as still water.
"""
import urllib.request

import numba as nb
import numpy as np
import xarray as xr

from common import RAW

CUR_PATH = RAW / "currents" / "drifter_monthlymeans.nc"
ERDDAP = ("https://erddap.aoml.noaa.gov/gdp/erddap/griddap/drifter_monthlymeans.nc?"
          "U%5B(0):1:(11)%5D%5B(-80.0):1:(130.0)%5D%5B(-62.0):1:(66.0)%5D,"
          "V%5B(0):1:(11)%5D%5B(-80.0):1:(130.0)%5D%5B(-62.0):1:(66.0)%5D")
KN_PER_MS = 1.943844


def fetch():
    if not CUR_PATH.exists():
        CUR_PATH.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(ERDDAP, headers={"User-Agent": "isochrones-of-sail"})
        with urllib.request.urlopen(req, timeout=600) as r, open(CUR_PATH, "wb") as f:
            f.write(r.read())
    return CUR_PATH


class Currents:
    def __init__(self, cu, cv, lat, lon):
        self.cu = np.ascontiguousarray(np.nan_to_num(cu), dtype=np.float32)  # (12, lat, lon), knots
        self.cv = np.ascontiguousarray(np.nan_to_num(cv), dtype=np.float32)
        self.lat = np.asarray(lat, float)
        self.lon = np.asarray(lon, float)

    @classmethod
    def load(cls):
        with xr.open_dataset(fetch()) as ds:
            u = ds.U.transpose("ClimatologicalMonth", "latitude", "longitude").values * KN_PER_MS
            v = ds.V.transpose("ClimatologicalMonth", "latitude", "longitude").values * KN_PER_MS
            return cls(u, v, ds.latitude.values, ds.longitude.values)

    @classmethod
    def uniform(cls, cu_kn, cv_kn):
        lat = np.arange(-70, 71, 2.0)
        lon = np.arange(-90, 141, 2.0)
        shape = (12, len(lat), len(lon))
        return cls(np.full(shape, cu_kn), np.full(shape, cv_kn), lat, lon)

    @property
    def params(self):
        return (self.cu, self.cv, float(self.lat[0]), float((self.lat[-1] - self.lat[0]) / (len(self.lat) - 1)),
                float(self.lon[0]), float((self.lon[-1] - self.lon[0]) / (len(self.lon) - 1)), 1)


DUMMY = (np.zeros((12, 2, 2), np.float32), np.zeros((12, 2, 2), np.float32), 0.0, 1.0, 0.0, 1.0, 0)


@nb.njit(cache=True, fastmath=True)
def current_at(cur, th, lat, lon):
    """(east, north) current in knots at time th (hours since 1800-01-01) and position."""
    cu, cv, lat0, dlat, lon0, dlon, on = cur
    if on == 0:
        return 0.0, 0.0
    nm, ny, nx = cu.shape
    yf = ((th / 24.0) % 365.2425) / 365.2425
    m = yf * 12.0 - 0.5  # 0 = mid January
    m0 = int(np.floor(m)) % nm
    m1 = (m0 + 1) % nm
    am = m - np.floor(m)
    fy = min(max((lat - lat0) / dlat, 0.0), ny - 1.001)
    fx = min(max((lon - lon0) / dlon, 0.0), nx - 1.001)
    iy, ix = int(fy), int(fx)
    ay, ax = fy - iy, fx - ix
    uu = 0.0
    vv = 0.0
    for mm, wm in ((m0, 1 - am), (m1, am)):
        for dy_, wy in ((0, 1 - ay), (1, ay)):
            for dx_, wx in ((0, 1 - ax), (1, ax)):
                w = wm * wy * wx
                uu += w * cu[mm, iy + dy_, ix + dx_]
                vv += w * cv[mm, iy + dy_, ix + dx_]
    return uu, vv


@nb.njit(cache=True, fastmath=True)
def ground_speed(v, hdg, cu, cv, v_min):
    """Speed over the ground along course hdg for a ship making v through the water.

    The ship steers into the cross-track current to hold its course, so it makes good
    sqrt(v^2 - c_cross^2) + c_along.
    """
    h = np.radians(hdg)
    sh = np.sin(h)
    ch = np.cos(h)
    along = cu * sh + cv * ch
    cross = cu * ch - cv * sh
    w = v * v - cross * cross
    g = along + (np.sqrt(w) if w > 0.0 else 0.0)
    return max(g, v_min)
