"""Download 20CRv3 10 m winds year by year, cut the routing domain, keep a small copy.

Source: NOAA-CIRES-DOE Twentieth Century Reanalysis v3 (NCAR GDEX d131003), served
openly through the OSDF mirror. Each raw yearly file is 0.3-0.8 GB, so the raw file
is deleted as soon as the subset is written (D: has little free space).

Kept per year:
  uv_YYYY.nc      ensemble-mean u10, v10, 6-hourly, every 2nd native point (~1.4 deg),
                  int16 with 0.01 m/s scale.
  spread_YYYY.nc  daily mean ensemble spread of the vector wind (m/s), every 4th point.

Usage: python src/fetch_winds.py [first_year last_year]
"""
import sys
import time
import urllib.request

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from common import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW, WINDS, YEAR_MAX, YEAR_MIN

BASE = "https://osdf-director.osg-htc.org/ncar/rda/d131003/anl"
TMP = RAW / "20crv3" / "tmp"
TMP.mkdir(parents=True, exist_ok=True)


def download(name):
    dest = TMP / name
    if dest.exists():
        return dest
    part = dest.with_suffix(".part")
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{BASE}/{name}", headers={"User-Agent": "isochrones-of-sail"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
            part.rename(dest)
            return dest
        except Exception as e:  # network hiccups: retry with backoff
            print(f"  retry {attempt + 1} for {name}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed to download {name}")


def domain_index(lat, lon, step):
    """Indices for the domain, longitudes reordered to run -80..130 continuously."""
    lon180 = np.where(lon > 180, lon - 360, lon)
    ilat = np.where((lat >= LAT_MIN - 1) & (lat <= LAT_MAX + 1))[0][::step]
    ilat = ilat[np.argsort(lat[ilat])]  # file runs north to south; make ascending
    ilon_all = np.where((lon180 >= LON_MIN - 1) & (lon180 <= LON_MAX + 1))[0]
    ilon_all = ilon_all[np.argsort(lon180[ilon_all])]
    ilon = ilon_all[::step]
    return ilat, ilon, lon180


def read_field(path, tstep, sstep):
    """Read a whole yearly field (chunks span 365 time steps, so read everything once)."""
    with netCDF4.Dataset(path) as nc:
        name = [k for k, v in nc.variables.items() if v.ndim == 3][0]
        var = nc.variables[name]
        var.set_auto_mask(False)
        data = var[::tstep, :, :]
        lat = nc.variables["latitude"][:]
        lon = nc.variables["longitude"][:]
        tvar = nc.variables["time"]
        times = netCDF4.num2date(tvar[::tstep], tvar.units, only_use_cftime_datetimes=False,
                                 only_use_python_datetimes=True)
    ilat, ilon, lon180 = domain_index(np.asarray(lat), np.asarray(lon), sstep)
    sub = data[:, ilat][:, :, ilon]
    sub[sub > 9000] = np.nan
    return sub, np.asarray(lat)[ilat], lon180[ilon], pd.to_datetime(times)


def subset_mean(year):
    out = WINDS / f"uv_{year}.nc"
    if out.exists():
        return
    arrays = {}
    for var, short in (("UGRD", "u10"), ("VGRD", "v10")):
        f = download(f"anl_mean_{year}_{var}_10m.nc")
        sub, lat, lon, times = read_field(f, 2, 2)
        arrays[short] = xr.DataArray(sub, dims=("time", "latitude", "longitude"),
                                     coords={"time": times, "latitude": lat, "longitude": lon})
        f.unlink()
    ds_out = xr.Dataset(arrays)
    enc = {k: {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768, "zlib": True, "complevel": 4}
           for k in arrays}
    ds_out.to_netcdf(out, encoding=enc)
    print(f"  wrote {out.name} {out.stat().st_size / 1e6:.1f} MB", flush=True)


def subset_spread(year):
    out = WINDS / f"spread_{year}.nc"
    if out.exists():
        return
    parts = {}
    for var, short in (("UGRD", "u10"), ("VGRD", "v10")):
        f = download(f"anl_spread_{year}_{var}_10m.nc")
        sub, lat, lon, times = read_field(f, 1, 4)
        parts[short] = xr.DataArray(sub, dims=("time", "latitude", "longitude"),
                                    coords={"time": times, "latitude": lat, "longitude": lon})
        f.unlink()
    mag = np.sqrt(parts["u10"] ** 2 + parts["v10"] ** 2).resample(time="1D").mean()
    mag.name = "spread"
    mag.to_dataset().to_netcdf(out, encoding={"spread": {"dtype": "float32", "zlib": True, "complevel": 4}})
    print(f"  wrote {out.name} {out.stat().st_size / 1e6:.1f} MB", flush=True)


def main():
    y0, y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (YEAR_MIN, YEAR_MAX)
    for year in range(y0, y1 + 1):
        t = time.time()
        print(f"{year}", flush=True)
        subset_mean(year)
        subset_spread(year)
        print(f"  done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
