"""Rerun the K3 rediscovery check for four models (PLAN.md Addendum 3)."""
import json

from climatology import Clim
from common import DERIVED, RESULTS, SOUTH_LIMIT
from polar import Polar
from prepare import k3_rediscovery
from router import Graph

VARIANTS = [("", None), ("", SOUTH_LIMIT), ("_logged", None), ("_logged", SOUTH_LIMIT), ("_hull", SOUTH_LIMIT)]
NAMES = {"": "reanalysis polar", "_logged": "logged-wind polar", "_hull": "tacking-hull polar"}


def main():
    checks = json.loads((RESULTS / "checks.json").read_text())
    out = {}
    for variant, lim in VARIANTS:
        name = NAMES[variant] + (", 42S limit" if lim else ", no limit")
        pol = Polar.load(DERIVED / f"polar{variant}.json")
        clim = Clim.load(DERIVED / f"clim_speed{variant}.npz")
        r = k3_rediscovery(Graph(lat_limit=lim), pol, clim)
        tag = variant + ("_s42" if lim else "")
        out[tag or "main"] = dict(name=name, **r)
        print(name, r["n_ok"], [(m["month"], round(m["equator_lon"], 1), round(m["lat_at_60E"], 1))
                                for m in r["months"]], flush=True)
    checks["K3_variants"] = out
    order = ["_s42", "_logged_s42"]
    primary = next((t for t in order if out[t]["passed"]), "_s42")
    checks["primary"] = dict(tag=primary, k3_passed=bool(out[primary]["passed"]), name=out[primary]["name"])
    print("primary model:", checks["primary"])
    (RESULTS / "checks.json").write_text(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    main()
