"""Run the whole pipeline in order.

  fetch_winds   20CRv3 10 m winds 1836-1855 (slow, about 1 h; skips years already done)
  landmask      Natural Earth coast raster
  voyages       CLIWOC logbooks -> legs of daily fixes
  prepare       clean legs, fit polar, climatology, checks K3-K6
  k3_variants   K3 for the polar x southern-limit variants (Addendum 3)
  route_legs    route every leg three ways (10-15 min per model on 15 threads; ~50 min with currents)
  robust_logged exploratory: polar fitted to the logged winds (Addendum 2)
  robust_hull   exploratory: tacking-hull polar with a no-go zone (Addendum 4)
  robust_currents exploratory: drifter current climatology, polars refitted through the water (Addendum 6)
  stats         hypotheses H1-H3 and exploratory summaries
  figures       posters and charts
  docs_data     data for docs/index.html
"""
import runpy
import sys
import time
from pathlib import Path

STEPS = ["fetch_winds", "landmask", "voyages", "prepare", "robust_logged", "robust_hull", "k3_variants",
         "route_legs --south", "route_legs", "route_legs --logged --south", "route_legs --hull --south",
         "robust_currents", "route_legs --currents --south", "route_legs --hull --currents --south",
         "stats", "figures", "docs_data"]
SRC = Path(__file__).resolve().parent

if __name__ == "__main__":
    first = sys.argv[1] if len(sys.argv) > 1 else STEPS[0]
    for step in STEPS[STEPS.index(first):]:
        t = time.time()
        print(f"== {step}", flush=True)
        name, *args = step.split()
        sys.argv = [str(SRC / f"{name}.py"), *args]
        runpy.run_path(str(SRC / f"{name}.py"), run_name="__main__")
        print(f"   {step} took {time.time() - t:.0f} s", flush=True)
