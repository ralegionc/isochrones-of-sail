# Data

Nothing downloaded is kept in the repository. `python src/run_all.py` fetches it all again.

| What | Source | How it is fetched | Kept locally as |
|---|---|---|---|
| CLIWOC 2.1 logbooks | Steven Ottens' TSV of the CLIWOC 2.1 release, `https://media.githubusercontent.com/media/stvno/stvno.github.io/master/page/cliwoc/CLIWOC21.tsv` (167 MB) | downloaded by hand once (see below) | `data/raw/cliwoc/CLIWOC21.tsv` |
| 20th Century Reanalysis v3, 10 m wind, ensemble mean and spread, 1836-1855 | NCAR GDEX dataset d131003, open OSDF mirror `https://osdf-director.osg-htc.org/ncar/rda/d131003/anl/anl_{mean,spread}_YYYY_{UGRD,VGRD}_10m.nc` (0.3-0.8 GB per file, 80 files) | `src/fetch_winds.py` | the raw files are deleted after subsetting; only `data/derived/winds/uv_YYYY.nc` (~36 MB/yr) and `spread_YYYY.nc` (~4 MB/yr) are kept |
| NOAA Global Drifter Program monthly current climatology v3.10 (Addendum 6) | AOML ERDDAP `drifter_monthlymeans`, U and V, 80W-130E, 62S-66N, 0.25 degree (83 MB) | `src/currents.py` | `data/raw/currents/drifter_monthlymeans.nc` |
| Natural Earth 1:50m land | `https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip` | `src/landmask.py` | `data/raw/naturalearth/` |

To get the logbooks:

```bash
mkdir -p data/raw/cliwoc
curl -L -o data/raw/cliwoc/CLIWOC21.tsv https://media.githubusercontent.com/media/stvno/stvno.github.io/master/page/cliwoc/CLIWOC21.tsv
```

## Wind subset

`fetch_winds.py` keeps the domain 80W-130E, 62S-66N. Mean winds keep every second
native grid point (about 1.4 degrees) and every second time step (6-hourly), stored
as int16 at 0.01 m/s. Spread keeps every fourth point as a daily mean of the vector
wind spread. The reanalysis assimilates surface pressure only, so the logbook winds
are an independent check on it.

## Citations

* Slivinski, L. C., et al. (2019). Towards a more reliable historical reanalysis:
  Improvements for version 3 of the Twentieth Century Reanalysis system. QJRMS 145, 2876-2908.
  Data: NOAA-CIRES-DOE 20th Century Reanalysis V3, doi:10.5065/H93G-WS83.
* Garcia-Herrera, R., et al. (2005). CLIWOC: A climatological database for the world's
  oceans 1750-1854. Climatic Change 73, 1-12.
* Laurindo, L., Mariano, A. and Lumpkin, R. (2017). An improved near-surface velocity climatology
  for the global ocean from drifter observations. Deep-Sea Research I 124, 73-92.
* Natural Earth, public domain.
