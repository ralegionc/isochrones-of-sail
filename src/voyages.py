"""Turn CLIWOC 2.1 logbook rows into clean legs of daily noon positions.

A leg is a run of daily positions at sea: no anchored days, no gap over 3 days,
at least 15 days long. Leg ends are trimmed until they sit at least ~0.3 degree
from land, so the router never starts or finishes inside a harbour.
"""
import numpy as np
import pandas as pd

from common import (DERIVED, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW, YEAR_MAX, YEAR_MIN,
                    haversine_km)
from landmask import LandMask

TSV = RAW / "cliwoc" / "CLIWOC21.tsv"
COLS = ["YR", "MO", "DY", "HR", "latitude", "longitude", "D", "W", "WI", "ShipName", "Nationality",
        "ShipType", "VoyageFrom", "VoyageTo", "Anchored", "LogbookIdent", "VoyageIni", "PosCoastal"]
MAX_KN_OUTLIER = 12.0   # a day's run above this to both neighbours marks a bad fix
MAX_KN_SPLIT = 14.0     # a run above this between kept fixes splits the leg
MAX_GAP_DAYS = 3.0
MIN_LEG_DAYS = 15.0
MIN_FIXES = 12


def load_rows():
    df = pd.read_csv(TSV, sep="\t", usecols=COLS, low_memory=False)
    num = ["YR", "MO", "DY", "HR", "latitude", "longitude", "D", "W", "WI", "Anchored", "PosCoastal"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df.YR >= YEAR_MIN) & (df.YR <= YEAR_MAX)]
    df = df.dropna(subset=["YR", "MO", "DY", "latitude", "longitude", "LogbookIdent"])
    date = pd.to_datetime(dict(year=df.YR, month=df.MO, day=df.DY), errors="coerce")
    df = df[date.notna()].copy()
    date = date[date.notna()]
    # HR is the UTC hour x100; missing hours default to local apparent noon
    hr = (df.HR / 100.0).where(df.HR.notna() & (df.HR >= 0) & (df.HR < 2400),
                               (12.0 - df.longitude / 15.0) % 24.0)
    df["time"] = date + pd.to_timedelta(hr, unit="h")
    # local noon offset, used to keep the fix nearest local noon when a day has several rows
    local_hr = (hr + df.longitude / 15.0) % 24.0
    df["noon_dist"] = np.abs(local_hr - 12.0)
    df["log_wdir"] = df.D.where((df.D >= 1) & (df.D <= 360))
    df["log_wspd"] = (df.W / 10.0).where(df.W.notna() & (df.W >= 0) & (df.W < 300))
    df["Anchored"] = df.Anchored.fillna(0)
    return df


def drop_outliers(g):
    """Iteratively drop single fixes that imply impossible runs to both neighbours."""
    for _ in range(5):
        if len(g) < 3:
            return g
        lat, lon = g.latitude.values, g.longitude.values
        t = g.time.values.astype("datetime64[s]").astype(float) / 3600.0
        d = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) / 1.852
        dt = np.maximum(np.diff(t), 1.0)
        kn = d / dt
        bad = np.zeros(len(g), bool)
        bad[1:-1] = (kn[:-1] > MAX_KN_OUTLIER) & (kn[1:] > MAX_KN_OUTLIER)
        if not bad.any():
            return g
        g = g[~bad]
    return g


def split_legs(g):
    """Yield runs of consecutive at-sea fixes; any anchored day ends a run."""
    run = (g.Anchored == 1).cumsum().values
    sea = g.Anchored.values != 1
    for r in np.unique(run[sea]):
        h = g[sea & (run == r)]
        if len(h) < 2:
            continue
        lat, lon = h.latitude.values, h.longitude.values
        t = h.time.values.astype("datetime64[s]").astype(float) / 3600.0
        d = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) / 1.852
        dt = np.diff(t)
        brk = (dt > MAX_GAP_DAYS * 24) | (d / np.maximum(dt, 1.0) > MAX_KN_SPLIT)
        starts = np.r_[0, np.where(brk)[0] + 1]
        ends = np.r_[np.where(brk)[0] + 1, len(h)]
        for a, b in zip(starts, ends):
            yield h.iloc[a:b]


def build_legs():
    mask = LandMask()
    df = load_rows()
    in_dom = df.latitude.between(LAT_MIN, LAT_MAX) & df.longitude.between(LON_MIN, LON_MAX)
    df["in_domain"] = in_dom
    legs, fixes = [], []
    for ident, g in df.sort_values("time").groupby("LogbookIdent", sort=False):
        g = g.assign(day=g.time.dt.floor("D")).sort_values(["day", "noon_dist"])
        g = g.drop_duplicates("day", keep="first").sort_values("time")
        g = drop_outliers(g)
        for leg in split_legs(g):
            if not leg.in_domain.all():
                continue
            on_land = mask.is_land(leg.latitude.values, leg.longitude.values)
            if on_land.mean() > 0.10:
                continue
            leg = leg[~on_land]
            near = mask.is_near_land(leg.latitude.values, leg.longitude.values)
            if (~near).sum() < MIN_FIXES:
                continue
            first, last = np.argmax(~near), len(near) - 1 - np.argmax(~near[::-1])
            leg = leg.iloc[first:last + 1]
            days = (leg.time.iloc[-1] - leg.time.iloc[0]).total_seconds() / 86400
            if days < MIN_LEG_DAYS or len(leg) < MIN_FIXES:
                continue
            leg_id = f"{ident}_{len(legs):04d}"
            row0, row1 = leg.iloc[0], leg.iloc[-1]
            legs.append(dict(
                leg_id=leg_id, logbook=ident, ship=row0.ShipName, nationality=row0.Nationality,
                ship_type=row0.ShipType, voyage_from=row0.VoyageFrom, voyage_to=row0.VoyageTo,
                t0=row0.time, t1=row1.time, days=days, n_fix=len(leg),
                lat0=row0.latitude, lon0=row0.longitude, lat1=row1.latitude, lon1=row1.longitude,
                route=classify(row0.latitude, row0.longitude, row1.latitude, row1.longitude)))
            f = leg[["time", "latitude", "longitude", "log_wdir", "log_wspd"]].copy()
            f.insert(0, "leg_id", leg_id)
            fixes.append(f)
    legs = pd.DataFrame(legs)
    fixes = pd.concat(fixes, ignore_index=True)
    legs.to_csv(DERIVED / "legs.csv", index=False)
    fixes.to_csv(DERIVED / "fixes.csv", index=False)
    print(f"{len(legs)} legs, {len(fixes)} daily fixes")
    print(legs.route.value_counts().to_string())
    print(legs.days.describe().to_string())
    return legs, fixes


def classify(lat0, lon0, lat1, lon1):
    europe0 = lat0 > 35 and lon0 < 15
    europe1 = lat1 > 35 and lon1 < 15
    indies0 = lon0 > 85 and -15 < lat0 < 10
    indies1 = lon1 > 85 and -15 < lat1 < 10
    if europe0 and indies1:
        return "outbound"
    if indies0 and europe1:
        return "homeward"
    return "other"


def load_legs():
    legs = pd.read_csv(DERIVED / "legs.csv", parse_dates=["t0", "t1"])
    fixes = pd.read_csv(DERIVED / "fixes.csv", parse_dates=["time"])
    return legs, fixes


if __name__ == "__main__":
    build_legs()
