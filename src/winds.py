"""Load 20CRv3 wind subsets and interpolate them fast (numba)."""
import numba as nb
import numpy as np
import pandas as pd
import xarray as xr

from common import WINDS

EPOCH = np.datetime64("1800-01-01T00:00")


def hours(t):
    """Hours since 1800-01-01 for numpy/pandas datetimes."""
    return (np.asarray(t, dtype="datetime64[m]") - EPOCH).astype(np.float64) / 60.0


class WindCube:
    """6-hourly u10/v10 on a regular (approximately) lat/lon grid as float32 arrays."""

    def __init__(self, u, v, times, lat, lon):
        self.u = np.ascontiguousarray(u, dtype=np.float32)
        self.v = np.ascontiguousarray(v, dtype=np.float32)
        self.th0 = float(hours(times[0]))
        self.dth = float((hours(times[1]) - hours(times[0])))
        self.lat0 = float(lat[0])
        self.dlat = float((lat[-1] - lat[0]) / (len(lat) - 1))
        self.lon0 = float(lon[0])
        self.dlon = float((lon[-1] - lon[0]) / (len(lon) - 1))
        self.times = np.asarray(times)
        self.lat = np.asarray(lat)
        self.lon = np.asarray(lon)

    @property
    def params(self):
        return (self.u, self.v, self.th0, self.dth, self.lat0, self.dlat, self.lon0, self.dlon)

    @classmethod
    def load(cls, t_start, t_end):
        """Winds covering [t_start, t_end] (pads by a day at each end)."""
        t_start = pd.Timestamp(t_start) - pd.Timedelta(days=1)
        t_end = pd.Timestamp(t_end) + pd.Timedelta(days=1)
        parts = []
        for year in range(t_start.year, t_end.year + 1):
            f = WINDS / f"uv_{year}.nc"
            if not f.exists():
                raise FileNotFoundError(f)
            with xr.open_dataset(f) as ds:
                parts.append(ds.sel(time=slice(t_start, t_end)).load())
        ds = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        u = ds.u10.values
        v = ds.v10.values
        u = np.nan_to_num(u, nan=0.0)
        v = np.nan_to_num(v, nan=0.0)
        return cls(u, v, ds.time.values, ds.latitude.values, ds.longitude.values)

    @classmethod
    def load_all(cls, y0=None, y1=None):
        """Every downloaded year as one continuous cube (about 3.3 GB of float32 for 20 years)."""
        from common import YEAR_MAX, YEAR_MIN
        y0 = y0 or YEAR_MIN
        y1 = y1 or YEAR_MAX
        us, vs, ts = [], [], []
        lat = lon = None
        for year in range(y0, y1 + 1):
            with xr.open_dataset(WINDS / f"uv_{year}.nc") as ds:
                us.append(np.nan_to_num(ds.u10.values.astype(np.float32)))
                vs.append(np.nan_to_num(ds.v10.values.astype(np.float32)))
                ts.append(ds.time.values)
                lat, lon = ds.latitude.values, ds.longitude.values
        times = np.concatenate(ts)
        step = np.diff(times).astype("timedelta64[h]").astype(int)
        assert (step == 6).all(), "wind cube is not continuous 6-hourly"
        return cls(np.concatenate(us), np.concatenate(vs), times, lat, lon)

    @classmethod
    def uniform(cls, u0, v0, days=400):
        """Constant wind everywhere, for known-answer tests."""
        lat = np.arange(-70, 71, 2.0)
        lon = np.arange(-90, 141, 2.0)
        times = pd.date_range("1840-01-01", periods=days * 4, freq="6h").values
        u = np.full((len(times), len(lat), len(lon)), u0, np.float32)
        v = np.full((len(times), len(lat), len(lon)), v0, np.float32)
        return cls(u, v, times, lat, lon)


@nb.njit(cache=True, fastmath=True)
def interp_uv(u, v, th0, dth, lat0, dlat, lon0, dlon, th, lat, lon):
    """Trilinear interpolation of (u, v) at time th (hours since 1800) and position."""
    nt, ny, nx = u.shape
    ft = (th - th0) / dth
    fy = (lat - lat0) / dlat
    fx = (lon - lon0) / dlon
    if ft < 0.0:
        ft = 0.0
    if ft > nt - 1.001:
        ft = nt - 1.001
    if fy < 0.0:
        fy = 0.0
    if fy > ny - 1.001:
        fy = ny - 1.001
    if fx < 0.0:
        fx = 0.0
    if fx > nx - 1.001:
        fx = nx - 1.001
    it = int(ft)
    iy = int(fy)
    ix = int(fx)
    at = ft - it
    ay = fy - iy
    ax = fx - ix
    uu = 0.0
    vv = 0.0
    for dt_ in range(2):
        wt = at if dt_ == 1 else 1.0 - at
        for dy_ in range(2):
            wy = ay if dy_ == 1 else 1.0 - ay
            for dx_ in range(2):
                wx = ax if dx_ == 1 else 1.0 - ax
                w = wt * wy * wx
                uu += w * u[it + dt_, iy + dy_, ix + dx_]
                vv += w * v[it + dt_, iy + dy_, ix + dx_]
    return uu, vv
