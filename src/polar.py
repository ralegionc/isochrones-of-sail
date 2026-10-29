"""Empirical fleet polar: daily speed made good versus 20CR wind speed and wind angle.

Course made good over a day already folds in tacking, so the polar is a table of
speed made good, not speed through the water.
"""
import json

import numba as nb
import numpy as np
import pandas as pd

from common import DERIVED, KM_PER_NM, angle_diff, bearing_deg, haversine_km, wind_from_uv
from winds import hours, interp_uv

WS_EDGES = np.array([0, 2, 4, 6, 8, 10, 12, 14, 17, 22], float)
TWA_EDGES = np.arange(0, 181, 15.0)
POLAR_PATH = DERIVED / "polar.json"
MIN_BIN = 25


def segment_table(fixes, cube, year_shift=0, currents=None):
    """One row per consecutive pair of daily fixes, with 20CR wind averaged along it.

    year_shift != 0 samples the wind on the same calendar day in another year
    (placebo winds for the known-answer checks). With currents (Addendum 6), the mean
    current along the segment is taken off the ground run, so speed_kn and course are
    through the water.
    """
    from currents import DUMMY, current_at
    cur = currents.params if currents is not None else DUMMY
    rows = []
    p = cube.params
    for leg_id, g in fixes.groupby("leg_id", sort=False):
        t = g.time.values
        lat = g.latitude.values
        lon = g.longitude.values
        th = hours(t)
        if year_shift:
            th_w = hours((pd.DatetimeIndex(t) + pd.DateOffset(years=year_shift)).values)
        else:
            th_w = th
        for k in range(len(g) - 1):
            dt_h = th[k + 1] - th[k]
            if not (18.0 <= dt_h <= 30.0):
                continue
            d_nm = haversine_km(lat[k], lon[k], lat[k + 1], lon[k + 1]) / KM_PER_NM
            course = bearing_deg(lat[k], lon[k], lat[k + 1], lon[k + 1])
            us, vs = 0.0, 0.0
            for f in np.linspace(0.1, 0.9, 5):
                uu, vv = interp_uv(*p, th_w[k] + f * dt_h, lat[k] + f * (lat[k + 1] - lat[k]),
                                   lon[k] + f * (lon[k + 1] - lon[k]))
                us += uu / 5
                vs += vv / 5
            cus, cvs = 0.0, 0.0
            if currents is not None:
                for f in np.linspace(0.1, 0.9, 5):
                    cu_, cv_ = current_at(cur, th[k] + f * dt_h, lat[k] + f * (lat[k + 1] - lat[k]),
                                          lon[k] + f * (lon[k + 1] - lon[k]))
                    cus += cu_ / 5
                    cvs += cv_ / 5
                ge = d_nm / dt_h * np.sin(np.radians(course)) - cus
                gn = d_nm / dt_h * np.cos(np.radians(course)) - cvs
                course = (np.degrees(np.arctan2(ge, gn)) + 360.0) % 360.0
                d_nm = np.hypot(ge, gn) * dt_h
            wfrom, wspd = wind_from_uv(us, vs)
            rows.append((leg_id, t[k], t[k + 1], lat[k], lon[k], lat[k + 1], lon[k + 1], dt_h, d_nm,
                         d_nm / dt_h, course, us, vs, wspd, wfrom, angle_diff(course, wfrom),
                         g.log_wdir.values[k + 1], g.log_wspd.values[k + 1]))
    cols = ["leg_id", "t_a", "t_b", "lat_a", "lon_a", "lat_b", "lon_b", "dt_h", "dist_nm", "speed_kn",
            "course", "u", "v", "tws", "wind_from", "twa", "log_wdir", "log_wspd"]
    return pd.DataFrame(rows, columns=cols)


def fit(seg):
    """Binned medians of speed made good, gaps filled, made monotone in wind up to 12 m/s."""
    nws, nta = len(WS_EDGES) - 1, len(TWA_EDGES) - 1
    table = np.full((nws, nta), np.nan)
    count = np.zeros((nws, nta), int)
    iw = np.clip(np.digitize(seg.tws, WS_EDGES) - 1, 0, nws - 1)
    ia = np.clip(np.digitize(seg.twa, TWA_EDGES) - 1, 0, nta - 1)
    for a in range(nws):
        for b in range(nta):
            s = seg.speed_kn.values[(iw == a) & (ia == b)]
            count[a, b] = len(s)
            if len(s) >= MIN_BIN:
                table[a, b] = np.median(s)
    # fill empty bins from the nearest filled wind bin at the same angle
    for b in range(nta):
        col = table[:, b]
        ok = np.where(~np.isnan(col))[0]
        if len(ok) == 0:
            continue
        for a in range(nws):
            if np.isnan(col[a]):
                col[a] = col[ok[np.argmin(np.abs(ok - a))]]
    # light smoothing across angle, then monotone non-decreasing in wind up to 12 m/s
    sm = table.copy()
    for b in range(nta):
        lo, hi = max(b - 1, 0), min(b + 2, nta)
        sm[:, b] = np.nanmean(table[:, lo:hi], axis=1)
    ws_c = 0.5 * (WS_EDGES[1:] + WS_EDGES[:-1])
    for a in range(1, nws):
        if ws_c[a] <= 12:
            sm[a] = np.maximum(sm[a], sm[a - 1])
    return sm, count


def save(table, count, extra=None, path=POLAR_PATH):
    d = dict(ws_edges=WS_EDGES.tolist(), twa_edges=TWA_EDGES.tolist(), table=table.tolist(),
             count=count.tolist())
    if extra:
        d.update(extra)
    path.write_text(json.dumps(d, indent=1))


class Polar:
    def __init__(self, table, ws_edges=WS_EDGES, twa_edges=TWA_EDGES):
        self.table = np.ascontiguousarray(table, dtype=np.float64)
        self.ws = 0.5 * (np.asarray(ws_edges)[1:] + np.asarray(ws_edges)[:-1])
        self.twa = 0.5 * (np.asarray(twa_edges)[1:] + np.asarray(twa_edges)[:-1])
        # anchor the ends so interpolation covers 0..180 degrees
        self.twa = np.r_[0.0, self.twa, 180.0]
        self.table = np.ascontiguousarray(np.c_[self.table[:, :1], self.table, self.table[:, -1:]])

    @classmethod
    def load(cls, path=POLAR_PATH):
        d = json.loads(path.read_text())
        return cls(np.array(d["table"]), d["ws_edges"], d["twa_edges"])

    @property
    def params(self):
        return (self.table, self.ws, self.twa)

    def __call__(self, tws, twa):
        tws = np.atleast_1d(np.asarray(tws, float))
        twa = np.atleast_1d(np.asarray(twa, float))
        out = np.empty(np.broadcast(tws, twa).shape)
        it = np.nditer([tws, twa, out], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
        for a, b, o in it:
            o[...] = polar_speed(self.table, self.ws, self.twa, float(a), float(b))
        return out


@nb.njit(cache=True, fastmath=True)
def polar_speed(table, ws, twa, s, a):
    """Bilinear lookup, clamped at the table edges. Returns knots made good."""
    nw = ws.shape[0]
    na = twa.shape[0]
    if s <= ws[0]:
        iw, fw = 0, 0.0
        # below the first bin centre scale towards zero speed in no wind
        scale = s / ws[0] if ws[0] > 0 else 1.0
    elif s >= ws[nw - 1]:
        iw, fw = nw - 2, 1.0
        scale = 1.0
    else:
        iw = 0
        while ws[iw + 1] < s:
            iw += 1
        fw = (s - ws[iw]) / (ws[iw + 1] - ws[iw])
        scale = 1.0
    if a <= 0.0:
        ia, fa = 0, 0.0
    elif a >= 180.0:
        ia, fa = na - 2, 1.0
    else:
        ia = 0
        while twa[ia + 1] < a:
            ia += 1
        fa = (a - twa[ia]) / (twa[ia + 1] - twa[ia])
    v = ((1 - fw) * (1 - fa) * table[iw, ia] + fw * (1 - fa) * table[iw + 1, ia]
         + (1 - fw) * fa * table[iw, ia + 1] + fw * fa * table[iw + 1, ia + 1])
    return v * scale


def convex_speed(polar, tws, n=720):
    """Best speed made good in each direction when tacking/gybing is allowed (convex hull)."""
    ang = np.linspace(0, 360, n, endpoint=False)  # bearing relative to the wind's source
    twa = np.where(ang <= 180, ang, 360 - ang)
    r = polar(np.full(n, tws), twa)
    pts = np.c_[r * np.sin(np.radians(ang)), r * np.cos(np.radians(ang))]
    best = np.zeros(n)
    for i, th in enumerate(np.radians(ang)):
        d = np.array([np.sin(th), np.cos(th)])
        e = np.array([-d[1], d[0]])
        a = pts @ d
        q = pts @ e
        b = a[i]  # sailing straight along d
        P, N = q > 1e-12, q < -1e-12
        if P.any() and N.any():
            # time share w on a heading left of d and 1-w on one right of d, so that the
            # sideways parts cancel: w*q_p + (1-w)*q_n = 0
            qp, qn = q[P][:, None], q[N][None, :]
            w = -qn / (qp - qn)
            comb = w * a[P][:, None] + (1 - w) * a[N][None, :]
            b = max(b, comb.max())
        best[i] = b
    return ang, best
