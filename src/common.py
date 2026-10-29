"""Shared paths, domain and geodesy helpers."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
WINDS = DERIVED / "winds"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"

for _d in (RAW, DERIVED, WINDS, RESULTS, FIGURES, DOCS):
    _d.mkdir(parents=True, exist_ok=True)

# Routing domain: North Atlantic to the Java Sea.
LON_MIN, LON_MAX = -80.0, 130.0
LAT_MIN, LAT_MAX = -62.0, 66.0

# 20CRv3 years covered by the official release that overlap CLIWOC.
YEAR_MIN, YEAR_MAX = 1836, 1855

# Southern limit for routes (PLAN.md Addendum 3): the fleet's own envelope.
SOUTH_LIMIT = -42.0

EARTH_R_KM = 6371.0
KM_PER_NM = 1.852


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def wind_from_uv(u, v):
    """Meteorological wind direction (degrees the wind blows FROM) and speed."""
    spd = np.hypot(u, v)
    frm = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return frm, spd


def angle_diff(a, b):
    """Smallest absolute difference between two angles in degrees (0..180)."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d
