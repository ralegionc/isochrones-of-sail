"""Time-dependent shortest-path routing for sailing ships, and replay of a fixed track.

The ocean is a 0.25 degree grid. Each node links to 32 neighbours (all primitive
offsets up to 3 cells), so headings are resolved to about 11 degrees. Arrival
times grow outward from the start like the isochrones a racing navigator draws;
the arrival-time field itself is kept so the isochrones can be drawn.

Two speed models share one search:
  mode 0: real 20CR winds at the time the ship would be there, through the polar.
  mode 1: climatology, the expected speed made good per heading for that month.
"""
import heapq

import numba as nb
import numpy as np

from common import DERIVED, KM_PER_NM, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, bearing_deg, haversine_km
from currents import DUMMY as _DUMMY_CUR
from currents import current_at, ground_speed
from landmask import LandMask
from polar import polar_speed
from winds import interp_uv

RES = 0.25
NY = int(round((LAT_MAX - LAT_MIN) / RES)) + 1
NX = int(round((LON_MAX - LON_MIN) / RES)) + 1
OFFS = np.array([(di, dj) for di in range(-3, 4) for dj in range(-3, 4)
                 if (di, dj) != (0, 0) and np.gcd(abs(di), abs(dj)) == 1], dtype=np.int64)
GRAPH_PATH = DERIVED / "graph_025.npz"
V_MIN_KN = 0.3  # drifting floor so calms do not create infinite edges


def node_lat(i):
    return LAT_MIN + RES * i


def node_lon(j):
    return LON_MIN + RES * j


@nb.njit(cache=True)
def _edge_valid(land, lres, ocean, offs):
    ny, nx = ocean.shape
    no = offs.shape[0]
    valid = np.zeros((ny, nx, no), np.uint8)
    lny, lnx = land.shape
    for i in range(ny):
        for j in range(nx):
            if not ocean[i, j]:
                continue
            for k in range(no):
                ni = i + offs[k, 0]
                nj = j + offs[k, 1]
                if ni < 0 or nj < 0 or ni >= ny or nj >= nx or not ocean[ni, nj]:
                    continue
                lat_a = LAT_MIN + RES * i
                lon_a = LON_MIN + RES * j
                dlat = RES * offs[k, 0]
                dlon = RES * offs[k, 1]
                n = int(max(abs(dlat), abs(dlon)) / 0.02) + 2
                ok = True
                for s in range(n + 1):
                    f = s / n
                    li = int((lat_a + f * dlat - LAT_MIN) / lres)
                    lj = int((lon_a + f * dlon - LON_MIN) / lres)
                    if li < 0 or lj < 0 or li >= lny or lj >= lnx or land[li, lj]:
                        ok = False
                        break
                if ok:
                    valid[i, j, k] = 1
    return valid


class Graph:
    def __init__(self, lat_limit=None):
        """lat_limit: optional southern limit in degrees (routes may not go south of it)."""
        if GRAPH_PATH.exists():
            z = np.load(GRAPH_PATH)
            self.ocean, self.valid = z["ocean"], z["valid"]
        else:
            mask = LandMask()
            lats = node_lat(np.arange(NY))
            lons = node_lon(np.arange(NX))
            L, M = np.meshgrid(lats, lons, indexing="ij")
            self.ocean = ~mask.is_land(L, M)
            self.valid = _edge_valid(mask.land, mask.res, self.ocean, OFFS)
            np.savez_compressed(GRAPH_PATH, ocean=self.ocean, valid=self.valid)
        lats = node_lat(np.arange(NY))
        if lat_limit is not None:
            allowed = lats >= lat_limit
            self.valid = self.valid.copy()
            self.valid[~allowed] = 0
            for k, (di, dj) in enumerate(OFFS):
                rows = np.arange(NY)
                tgt = np.clip(rows + di, 0, NY - 1)
                self.valid[~allowed[tgt], :, k] = 0
            self.ocean = self.ocean & allowed[:, None]
        # per-row edge length (km) and bearing (deg) for each offset
        self.dist = np.zeros((NY, len(OFFS)))
        self.bear = np.zeros((NY, len(OFFS)))
        for k, (di, dj) in enumerate(OFFS):
            la2 = lats + RES * di
            self.dist[:, k] = haversine_km(lats, 0.0, la2, RES * dj)
            self.bear[:, k] = bearing_deg(lats, 0.0, la2, RES * dj)

    def snap(self, lat, lon):
        """Nearest ocean node that has outgoing edges."""
        i0 = int(round((lat - LAT_MIN) / RES))
        j0 = int(round((lon - LON_MIN) / RES))
        best, bd = None, 1e9
        for r in range(0, 12):
            for i in range(max(i0 - r, 0), min(i0 + r + 1, NY)):
                for j in range(max(j0 - r, 0), min(j0 + r + 1, NX)):
                    if self.ocean[i, j] and self.valid[i, j].any():
                        d = (i - i0) ** 2 + (j - j0) ** 2
                        if d < bd:
                            best, bd = (i, j), d
            if best is not None:
                return best
        raise ValueError(f"no ocean node near {lat}, {lon}")


@nb.njit(cache=True, fastmath=True)
def _speed_wind(wp, pp, th, lat, lon, heading):
    u, v, th0, dth, lat0, dlat, lon0, dlon = wp
    uu, vv = interp_uv(u, v, th0, dth, lat0, dlat, lon0, dlon, th, lat, lon)
    spd = np.sqrt(uu * uu + vv * vv)
    wfrom = (np.degrees(np.arctan2(-uu, -vv)) + 360.0) % 360.0
    twa = abs((heading - wfrom + 180.0) % 360.0 - 180.0)
    s = polar_speed(pp[0], pp[1], pp[2], spd, twa)
    return max(s, V_MIN_KN)


@nb.njit(cache=True, fastmath=True)
def _speed_clim(cp, th_rel, lat, lon, heading):
    """Expected speed made good from the monthly climatology table."""
    tab, mf, mf_dt, lat0, dlat, lon0, dlon = cp
    nm, ny, nx, nh = tab.shape
    k = int(th_rel / mf_dt)
    if k >= mf.shape[0] - 1:
        k = mf.shape[0] - 2
    if k < 0:
        k = 0
    m = mf[k]  # fractional month index, 0 = mid January
    m0 = int(np.floor(m)) % nm
    m1 = (m0 + 1) % nm
    am = m - np.floor(m)
    fy = min(max((lat - lat0) / dlat, 0.0), ny - 1.001)
    fx = min(max((lon - lon0) / dlon, 0.0), nx - 1.001)
    iy, ix = int(fy), int(fx)
    ay, ax = fy - iy, fx - ix
    fh = (heading % 360.0) / (360.0 / nh)
    ih0 = int(fh) % nh
    ih1 = (ih0 + 1) % nh
    ah = fh - int(fh)
    s = 0.0
    for mm, wm in ((m0, 1 - am), (m1, am)):
        for dy_, wy in ((0, 1 - ay), (1, ay)):
            for dx_, wx in ((0, 1 - ax), (1, ax)):
                w = wm * wy * wx
                s += w * ((1 - ah) * tab[mm, iy + dy_, ix + dx_, ih0] + ah * tab[mm, iy + dy_, ix + dx_, ih1])
    return max(s, V_MIN_KN)


@nb.njit(cache=True, nogil=True)
def _dijkstra(valid, dist, bear, offs, si, sj, ti, tj, th_start, t_max_h, mode, wp, pp, cp, cur, full):
    ny, nx, no = valid.shape
    T = np.full(ny * nx, np.inf)
    pred = np.full(ny * nx, -1, np.int64)
    s = si * nx + sj
    tgt = ti * nx + tj
    T[s] = 0.0
    heap = [(0.0, s)]
    while len(heap) > 0:
        t, idx = heapq.heappop(heap)
        if t > T[idx]:
            continue
        if idx == tgt and not full:
            break
        if t > t_max_h:
            break
        i = idx // nx
        j = idx % nx
        lat = LAT_MIN + RES * i
        lon = LON_MIN + RES * j
        for k in range(no):
            if valid[i, j, k] == 0:
                continue
            ni = i + offs[k, 0]
            nj = j + offs[k, 1]
            hdg = bear[i, k]
            # time the edge exactly as replay() times a track: ~20 km sub-steps, wind at each midpoint
            m = max(int(dist[i, k] / 20.0), 1)
            step_nm = dist[i, k] / m / KM_PER_NM
            tn = t
            for s in range(m):
                f = (s + 0.5) / m
                plat = lat + f * RES * offs[k, 0]
                plon = lon + f * RES * offs[k, 1]
                if mode == 0:
                    v = _speed_wind(wp, pp, th_start + tn, plat, plon, hdg)
                else:
                    v = _speed_clim(cp, tn, plat, plon, hdg)
                if cur[6] != 0:
                    cu, cv = current_at(cur, th_start + tn, plat, plon)
                    v = ground_speed(v, hdg, cu, cv, V_MIN_KN)
                tn += step_nm / v
            nidx = ni * nx + nj
            if tn < T[nidx]:
                T[nidx] = tn
                pred[nidx] = idx
                heapq.heappush(heap, (tn, nidx))
    return T, pred


def _path(pred, si, sj, ti, tj):
    idx = ti * NX + tj
    out = []
    while idx != -1:
        out.append(idx)
        if idx == si * NX + sj:
            break
        idx = pred[idx]
    out = np.array(out[::-1])
    if out[0] != si * NX + sj:
        return None
    return node_lat(out // NX), node_lon(out % NX)


_DUMMY_WIND = (np.zeros((2, 2, 2), np.float32), np.zeros((2, 2, 2), np.float32), 0.0, 6.0, 0.0, 1.0, 0.0, 1.0)
_DUMMY_CLIM = (np.ones((12, 2, 2, 36), np.float32), np.zeros(4), 6.0, 0.0, 1.0, 0.0, 1.0)


def route(graph, lat0, lon0, lat1, lon1, th_start, t_max_h, polar=None, cube=None, clim=None,
          full=False, currents=None):
    """Fastest path from (lat0, lon0) to (lat1, lon1) leaving at th_start (hours since 1800).

    currents: optional Currents climatology (PLAN.md Addendum 6); speeds are then over the ground.
    Returns dict with the path, the search's own arrival time (h) and the arrival field.
    """
    si, sj = graph.snap(lat0, lon0)
    ti, tj = graph.snap(lat1, lon1)
    if clim is None:
        mode, wp, cp = 0, cube.params, _DUMMY_CLIM
    else:
        mode, wp, cp = 1, _DUMMY_WIND, clim.params_for(th_start, t_max_h)
    pp = polar.params
    cur = currents.params if currents is not None else _DUMMY_CUR
    T, pred = _dijkstra(graph.valid, graph.dist, graph.bear, OFFS, si, sj, ti, tj, float(th_start),
                        float(t_max_h), mode, wp, pp, cp, cur, full)
    path = _path(pred, si, sj, ti, tj)
    return dict(path=path, t_h=T[ti * NX + tj], field=T.reshape(NY, NX))


@nb.njit(cache=True, fastmath=True, nogil=True)
def _replay(lats, lons, th_start, wp, pp, cur, step_km):
    n = lats.shape[0]
    arr = np.zeros(n)
    t = 0.0
    for k in range(n - 1):
        la, lo, lb, lob = lats[k], lons[k], lats[k + 1], lons[k + 1]
        p1, p2 = np.radians(la), np.radians(lb)
        dl = np.radians(lob - lo)
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        d_km = 2 * 6371.0 * np.arcsin(np.sqrt(min(max(a, 0.0), 1.0)))
        x = np.sin(dl) * np.cos(p2)
        y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
        hdg = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
        m = max(int(d_km / step_km), 1)
        for s in range(m):
            f = (s + 0.5) / m
            plat = la + f * (lb - la)
            plon = lo + f * (lob - lo)
            v = _speed_wind(wp, pp, th_start + t, plat, plon, hdg)
            if cur[6] != 0:
                cu, cv = current_at(cur, th_start + t, plat, plon)
                v = ground_speed(v, hdg, cu, cv, V_MIN_KN)
            t += (d_km / m) / KM_PER_NM / v
        arr[k + 1] = t
    return arr


def replay(lats, lons, th_start, cube, polar, step_km=20.0, currents=None):
    """Sail a fixed track through the winds (and currents, if given); returns hours to each vertex."""
    cur = currents.params if currents is not None else _DUMMY_CUR
    return _replay(np.asarray(lats, float), np.asarray(lons, float), float(th_start), cube.params,
                   polar.params, cur, float(step_km))
