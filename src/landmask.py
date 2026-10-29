"""Rasterise Natural Earth 1:50m land onto a 0.05 degree grid over the routing domain."""
import io
import urllib.request
import zipfile

import numpy as np
import shapefile  # pyshp
from PIL import Image, ImageDraw
from scipy import ndimage

from common import DERIVED, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW

RES = 0.05
NE_URL = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
NE_DIR = RAW / "naturalearth"
MASK_PATH = DERIVED / "land_005.npz"


def fetch():
    NE_DIR.mkdir(parents=True, exist_ok=True)
    shp = NE_DIR / "ne_50m_land.shp"
    if not shp.exists():
        req = urllib.request.Request(NE_URL, headers={"User-Agent": "isochrones-of-sail"})
        data = urllib.request.urlopen(req, timeout=120).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(NE_DIR)
    return shp


def build():
    shp = fetch()
    nx = int(round((LON_MAX - LON_MIN) / RES))
    ny = int(round((LAT_MAX - LAT_MIN) / RES))
    img = Image.new("1", (nx, ny), 0)
    draw = ImageDraw.Draw(img)
    reader = shapefile.Reader(str(shp))
    for s in reader.shapes():
        pts = np.asarray(s.points)
        parts = list(s.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ring = pts[a:b]
            if ring[:, 0].max() < LON_MIN - 1 or ring[:, 0].min() > LON_MAX + 1:
                continue
            px = (ring[:, 0] - LON_MIN) / RES
            py = (LAT_MAX - ring[:, 1]) / RES  # image row 0 = north
            draw.polygon(list(zip(px, py)), fill=1)
    land = np.array(img, dtype=bool)[::-1]  # row 0 = south
    lats = LAT_MIN + RES * (np.arange(ny) + 0.5)
    lons = LON_MIN + RES * (np.arange(nx) + 0.5)
    # "near land": within 0.3 degree (about 20-33 km)
    near = ndimage.binary_dilation(land, structure=np.ones((13, 13), bool))
    np.savez_compressed(MASK_PATH, land=land, near=near, lats=lats, lons=lons, res=RES)
    print(f"land fraction {land.mean():.3f}, saved {MASK_PATH.name}")


class LandMask:
    def __init__(self):
        if not MASK_PATH.exists():
            build()
        z = np.load(MASK_PATH)
        self.land = z["land"]
        self.near = z["near"]
        self.res = float(z["res"])

    def _idx(self, lat, lon):
        i = np.clip(((np.asarray(lat) - LAT_MIN) / self.res).astype(int), 0, self.land.shape[0] - 1)
        j = np.clip(((np.asarray(lon) - LON_MIN) / self.res).astype(int), 0, self.land.shape[1] - 1)
        return i, j

    def is_land(self, lat, lon):
        i, j = self._idx(lat, lon)
        return self.land[i, j]

    def is_near_land(self, lat, lon):
        i, j = self._idx(lat, lon)
        return self.near[i, j]


if __name__ == "__main__":
    build()
