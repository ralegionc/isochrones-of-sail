"""Night-chart posters and charts for Isochrones of Sail."""
import gzip
import json

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

matplotlib.use("Agg")

from common import (DERIVED, FIGURES, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW, RESULTS,  # noqa: E402
                    SOUTH_LIMIT)

BG = "#0a0d12"
LAND = "#151b26"
COAST = "#2a3446"
TEXT = "#e8e4d8"
SUBTLE = "#8a93a3"
AMBER = "#f2a541"     # the logged track
WHITE = "#f4f1e8"     # hindsight optimum
CYAN = "#5ec8e5"      # climatology route
FONT = "DejaVu Sans"
PC = ccrs.PlateCarree()
SHOW = "_hull_s42"   # maps use the one model that passes the rediscovery check (labelled as exploratory)
PRIMARY = "_s42"
ISO = LinearSegmentedColormap.from_list("iso", ["#173a5e", "#1f6f8b", "#3fb5a3", "#b9d66b", "#f2c14e", "#f28a3a"])

plt.rcParams.update({"font.family": FONT, "text.color": TEXT, "axes.labelcolor": SUBTLE,
                     "xtick.color": SUBTLE, "ytick.color": SUBTLE, "axes.edgecolor": COAST})


def land():
    g = gpd.read_file(RAW / "naturalearth" / "ne_50m_land.shp")
    return g.clip((LON_MIN - 5, LAT_MIN - 5, LON_MAX + 5, LAT_MAX + 5))


def map_axes(fig, rect, extent=(LON_MIN, LON_MAX, LAT_MIN + 4, LAT_MAX - 4), central=25):
    ax = fig.add_axes(rect, projection=ccrs.EqualEarth(central_longitude=central))
    ax.set_extent(extent, crs=PC)
    ax.set_facecolor(BG)
    ax.spines["geo"].set_visible(False)
    ax.add_geometries(land().geometry, crs=PC, facecolor=LAND, edgecolor=COAST, linewidth=0.4, zorder=2)
    return ax


def glow_line(ax, lons, lats, color, lw=1.2, alpha=1.0, z=5, glow=True, ls="-"):
    if glow:
        for w, a in ((lw * 5, 0.06), (lw * 2.8, 0.12)):
            ax.plot(lons, lats, color=color, lw=w, alpha=a * alpha, transform=PC, zorder=z, solid_capstyle="round")
    ax.plot(lons, lats, color=color, lw=lw, alpha=alpha, transform=PC, zorder=z + 0.1, ls=ls,
            solid_capstyle="round")


def lines(ax, paths, color, lw=0.5, alpha=0.2, z=4):
    segs = [np.c_[p[1], p[0]] for p in paths if len(p[0]) > 1]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, transform=PC, zorder=z))


def load(tag=SHOW):
    res = pd.read_csv(RESULTS / f"legs_routed{tag}.csv", parse_dates=["t0", "t1"])
    with gzip.open(RESULTS / f"paths{tag}.json.gz", "rt") as f:
        paths = json.load(f)
    summary = json.loads((RESULTS / "summary.json").read_text())
    checks = json.loads((RESULTS / "checks.json").read_text())
    return res, paths, summary, checks


def pick_showcase(res):
    ob = res[(res.route == "outbound") & (res.n_fix >= 0.9 * res.t_act_h / 24)].copy()
    ob = ob[ob.lat0 > 44]
    med = res[res.route == "outbound"].R_hind.median()
    ob["score"] = (ob.R_hind - med).abs()
    return ob.sort_values("score").iloc[0]


def hero(res, paths):
    from polar import Polar
    from router import NX, NY, RES, node_lat, node_lon, route, Graph
    from winds import WindCube, hours

    leg = pick_showcase(res)
    p = paths[leg.leg_id]
    act_lat, act_lon = np.array(p["act"][0]), np.array(p["act"][1])
    t_act = leg.t_act_h
    cube = WindCube.load(leg.t0, leg.t0 + pd.Timedelta(hours=2.2 * t_act + 24 * 40))
    pol = Polar.load(DERIVED / f"polar{SHOW.replace('_s42', '')}.json")
    th0 = float(hours(np.datetime64(leg.t0)))
    out = route(Graph(lat_limit=SOUTH_LIMIT), act_lat[0], act_lon[0], act_lat[-1], act_lon[-1], th0,
                1.25 * t_act, polar=pol, cube=cube, full=True)
    field = out["field"] / 24.0
    lat_g = node_lat(np.arange(NY))
    lon_g = node_lon(np.arange(NX))
    field = np.where(np.isfinite(field), field, np.nan)
    np.savez_compressed(DERIVED / "hero_field.npz", field=field.astype(np.float32), lat=lat_g, lon=lon_g,
                        leg_id=leg.leg_id)

    fig = plt.figure(figsize=(13, 11.6), dpi=250, facecolor=BG)
    ax = map_axes(fig, [0.02, 0.07, 0.96, 0.78], extent=(-46, 112, -49, 56), central=33)
    # mean wind over the voyage as faint streaks
    sel = slice(0, int(t_act / 6))
    u = cube.u[sel].mean(0)
    v = cube.v[sel].mean(0)
    ax.streamplot(cube.lon, cube.lat, u, v, transform=PC, color="#3a4a63", density=2.4, linewidth=0.45,
                  arrowsize=0.5, zorder=1)
    days = np.nanmax(field[np.isfinite(field)]) if np.isfinite(field).any() else 100
    levels = np.arange(5, min(days, 1.2 * t_act / 24), 5)
    sub = (slice(None, None, 2), slice(None, None, 2))
    for w, a in ((4.0, 0.07), (2.0, 0.15)):
        ax.contour(lon_g[sub[1]], lat_g[sub[0]], field[sub], levels=levels, cmap=ISO, linewidths=w, alpha=a,
                   transform=PC, zorder=3)
    cs = ax.contour(lon_g[sub[1]], lat_g[sub[0]], field[sub], levels=levels, cmap=ISO, linewidths=0.8,
                    transform=PC, zorder=3.1)
    lab = [lv for lv in levels if lv % 20 == 0]
    txt = ax.clabel(cs, levels=lab, fmt="%d d", fontsize=7, colors=TEXT)
    for t in txt:
        t.set_path_effects([pe.withStroke(linewidth=2, foreground=BG)])
    glow_line(ax, p["clim"][1], p["clim"][0], CYAN, lw=1.1, z=6, ls=(0, (4, 3)), glow=False)
    glow_line(ax, p["opt"][1], p["opt"][0], WHITE, lw=1.5, z=7)
    glow_line(ax, act_lon, act_lat, AMBER, lw=1.6, z=8)
    ax.scatter(act_lon[::5], act_lat[::5], s=7, color=AMBER, transform=PC, zorder=9, lw=0)
    ship = str(leg.ship).title()
    fig.text(0.04, 0.95, "Isochrones of Sail", fontsize=30, fontweight="bold")
    fig.text(0.04, 0.918, f"The {ship}, {str(leg.voyage_from).title()} to {str(leg.voyage_to).title()}, "
             f"left the Channel approaches on {leg.t0:%d %B %Y}", fontsize=13, color=SUBTLE)
    fig.text(0.04, 0.895, "Each ring marks how far the fastest possible ship could have got, every 5 days, in the winds "
             "of those exact days" + chr(10) + "(20th Century Reanalysis v3).", fontsize=10.5, color=SUBTLE, va="top")
    handles = [Line2D([], [], color=AMBER, lw=2, label=f"Logged track: {t_act / 24:.0f} days"),
               Line2D([], [], color=WHITE, lw=2, label=f"Hindsight optimum: {leg.t_opt_h / 24 * t_act / leg.t_rep_h:.0f} days"),
               Line2D([], [], color=CYAN, lw=1.5, ls=(0, (4, 3)),
                      label=f"Best route from wind charts alone: {leg.t_clim_h / 24 * t_act / leg.t_rep_h:.0f} days")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=10, labelcolor=TEXT,
              bbox_to_anchor=(0.0, 0.02))
    fig.text(0.04, 0.03, "Model: fleet polar with a square-rigger's no-go zone (tacking at 67.5 degrees), routes kept "
             "north of 42S. Ghost times are put on this voyage's logged clock. Track: CLIWOC 2.1. Coast: Natural Earth.",
             fontsize=7.5, color=SUBTLE)
    fig.savefig(FIGURES / "hero_isochrones.png", facecolor=BG, pad_inches=0.2)
    plt.close(fig)
    return leg.leg_id


def fleet(res, paths):
    fig = plt.figure(figsize=(16, 9.5), dpi=250, facecolor=BG)
    for k, (route, title) in enumerate((("outbound", "Outbound to the Indies"), ("homeward", "Homeward to Europe"))):
        ax = map_axes(fig, [0.01 + 0.495 * k, 0.05, 0.48, 0.78], extent=(-50, 115, -50, 58), central=30)
        sub = res[res.route == route]
        lines(ax, [paths[i]["opt"] for i in sub.leg_id], WHITE, lw=0.5, alpha=0.18)
        lines(ax, [paths[i]["act"][:2] for i in sub.leg_id], AMBER, lw=0.55, alpha=0.28, z=5)
        ax.set_title(f"{title}  ({len(sub)} voyages)", color=TEXT, fontsize=13, loc="left")
    fig.text(0.02, 0.94, "The road they took, and the road the winds allowed", fontsize=24, fontweight="bold")
    fig.text(0.02, 0.905, "Amber: logged tracks of Dutch East Indiamen, 1836-1855.  White: the hindsight-fastest "
             "route for each voyage, through the winds of its own days (no-go polar, north of 42S).", fontsize=11,
             color=SUBTLE)
    fig.savefig(FIGURES / "fleet.png", facecolor=BG, pad_inches=0.2)
    plt.close(fig)


def rediscovery(res, paths, checks):
    fig = plt.figure(figsize=(17, 8.6), dpi=230, facecolor=BG)
    ob = res[res.route == "outbound"]
    cm = LinearSegmentedColormap.from_list("months", ["#5ec8e5", "#7be0a8", "#e8e36b", "#f2a541", "#f2786b",
                                                       "#d68cf2", "#8fa2f5", "#5ec8e5"])
    for k, tag in enumerate((PRIMARY, SHOW)):
        v = checks["K3_variants"][tag]
        ax = map_axes(fig, [0.01 + 0.495 * k, 0.06, 0.48, 0.74], extent=(-50, 115, -50, 55), central=30)
        lines(ax, [paths[i]["act"][:2] for i in ob.leg_id], AMBER, lw=0.4, alpha=0.12, z=3)
        for r in v["months"]:
            la, lo = r["path"]
            glow_line(ax, lo, la, cm((r["month"] - 0.5) / 12), lw=1.1, z=6)
        ax.plot([-60, 120], [0, 0], color=SUBTLE, lw=0.4, ls=":", transform=PC, zorder=2)
        verdict = "passes" if v["passed"] else "fails"
        ax.set_title(f"{v['name']}: {v['n_ok']} of 12 months, {verdict}", color=TEXT, fontsize=12, loc="left")
    fig.text(0.02, 0.94, "Rediscovered from the winds alone?", fontsize=24, fontweight="bold")
    fig.text(0.02, 0.905, "The fastest route from the Channel to Java Head for each departure month, planned from "
             "1836-1855 average winds only. Faint amber: the logged outbound tracks.", fontsize=11, color=SUBTLE)
    fig.text(0.02, 0.88, "Pre-registered test: cross the equator between 15W and 35W (the volta do mar) in at least 10 of "
             "12 months. Giving the polar a square-rigger's no-go zone is what moves the routes west.",
             fontsize=10, color=SUBTLE)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 12))
    cax = fig.add_axes([0.76, 0.08, 0.2, 0.013])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=np.arange(0.5, 12, 1))
    cb.ax.set_xticklabels(list("JFMAMJJASOND"), fontsize=8, color=TEXT)
    cb.outline.set_visible(False)
    cb.set_label("departure month", color=SUBTLE, fontsize=8)
    fig.savefig(FIGURES / "rediscovery.png", facecolor=BG, pad_inches=0.2)
    plt.close(fig)


def dark_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)


def ratios(summary):
    models = summary["models"]
    order = [t for t in ("_s42", "_hull_s42", "none", "_logged_s42", "_cur_s42", "_hull_cur_s42") if t in models]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=220, facecolor=BG,
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    dark_ax(ax)
    for k, t in enumerate(order[::-1]):
        m = models[t]
        for key, c, dy in (("H1", WHITE, 0.12), ("H2", CYAN, -0.12)):
            lo, hi = m[key]["ci95"]
            ax.plot([lo, hi], [k + dy] * 2, color=c, lw=2.2, solid_capstyle="round")
            ax.scatter([m[key]["median"]], [k + dy], color=c, s=36, zorder=3)
        label = m["name"] + ("" if m["k3_passed"] else "  [fails K3]")
        ax.text(0.995, k + 0.32, label, color=TEXT if m["k3_passed"] else SUBTLE, fontsize=8.5, ha="right",
                transform=ax.get_yaxis_transform())
    ax.axvline(1.0, color=SUBTLE, lw=0.8, ls="--")
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.set_xlabel("logged route time / ghost route time (median, 95% CI over ships)")
    ax.legend(handles=[Line2D([], [], color=WHITE, lw=2.2, label="vs hindsight optimum (R_hind)"),
                       Line2D([], [], color=CYAN, lw=2.2, label="vs wind-chart route (R_clim)")],
              frameon=False, labelcolor=TEXT, fontsize=9, loc="lower left")
    ax.set_title("How much faster the ghost ships were", color=TEXT, fontsize=11, loc="left")
    ax = axes[1]
    dark_ax(ax)
    for t, c in ((PRIMARY, SUBTLE), (SHOW, AMBER)):
        res = pd.read_csv(RESULTS / f"legs_routed{t}.csv")
        ax.scatter(res.year, res.R_clim, s=5, color=c, alpha=0.3, lw=0)
        roll = res.sort_values("year").set_index("year").R_clim.rolling(80, center=True, min_periods=40).median()
        h3 = models[t]["H3"]
        name = models[t]["name"].split(" (")[0]
        ax.plot(roll.index, roll.values, color=c, lw=2.2,
                label=f"{name}: rho {h3['spearman_rho']:.2f}, p = {h3['p']:.2f}")
    ax.axvline(1848, color=CYAN, lw=1, ls=":")
    ax.text(1848.2, 0.66, "Maury's first wind charts", color=CYAN, fontsize=8, va="bottom")
    ax.axhline(1.0, color=SUBTLE, lw=0.8, ls="--")
    ax.set_ylim(0.6, 2.0)
    ax.set_xlabel("departure year")
    ax.set_ylabel("R_clim")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8.5, loc="upper left")
    ax.set_title("Did route choice improve over time? (H3)", color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "ratios.png", facecolor=BG)
    plt.close(fig)


def equator():
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=220, facecolor=BG)
    dark_ax(ax)
    bins = np.arange(-40, 6, 1.5)
    a = pd.read_csv(RESULTS / f"legs_routed{SHOW}.csv")
    b = pd.read_csv(RESULTS / f"legs_routed{PRIMARY}.csv")
    a, b = a[a.route == "outbound"], b[b.route == "outbound"]
    for x, c, name, ls in ((a.eq_lon_act, AMBER, "logged", "-"),
                           (a.eq_lon_clim, CYAN, "chart route, no-go polar", "-"),
                           (b.eq_lon_clim, SUBTLE, "chart route, blurred polar", "--")):
        x = x.dropna()
        ax.hist(x, bins=bins, histtype="step", lw=1.8, color=c, ls=ls, label=f"{name} (median {x.median():.1f})")
    ax.set_xlabel("longitude where the route crosses the equator (negative = west)")
    ax.set_ylabel("outbound voyages")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper left")
    ax.set_title("Where to cross the line", color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "equator.png", facecolor=BG)
    plt.close(fig)


def currents_fig(summary):
    """Addendum 6: homeward voyages with and without ocean currents (chart routes vs logged)."""
    from currents import Currents
    cur = Currents.load()
    models = summary["models"]
    fig = plt.figure(figsize=(15, 9.6), dpi=230, facecolor=BG)
    for k, tag in enumerate(("_hull_s42", "_hull_cur_s42")):
        res = pd.read_csv(RESULTS / f"legs_routed{tag}.csv")
        with gzip.open(RESULTS / f"paths{tag}.json.gz", "rt") as f:
            paths = json.load(f)
        ax = map_axes(fig, [0.01 + 0.5 * k, 0.05, 0.48, 0.76], extent=(-62, 22, -38, 56), central=-20)
        if k == 1:
            u = cur.cu.mean(0)[::4, ::4]
            v = cur.cv.mean(0)[::4, ::4]
            ax.streamplot(cur.lon[::4], cur.lat[::4], u, v, transform=PC, color="#2f6f8f", density=2.6,
                          linewidth=0.5, arrowsize=0.5, zorder=1)
        hw = res[res.route == "homeward"]
        lines(ax, [paths[i]["clim"] for i in hw.leg_id], CYAN, lw=0.6, alpha=0.3, z=4)
        lines(ax, [paths[i]["act"][:2] for i in hw.leg_id], AMBER, lw=0.6, alpha=0.3, z=5)
        ax.plot([-70, 30], [30, 30], color=SUBTLE, lw=0.5, ls=":", transform=PC, zorder=3)
        h = models[tag]["homeward_30N"]
        label = "with currents" if k else "no currents"
        ax.set_title(f"{label}: cross 30N at {abs(h['clim_median']):.0f}W (chart route) vs "
                     f"{abs(h['act_median']):.0f}W (logged)", color=TEXT, fontsize=11.5, loc="left")
    fig.text(0.02, 0.94, "Do currents send them home round the Azores?", fontsize=24, fontweight="bold")
    fig.text(0.02, 0.905, "Homeward voyages. Amber: logged tracks. Cyan: the route planned from wind charts, "
             "tacking-hull polar, north of 42S. Right: with the drifter current climatology (streamlines).",
             fontsize=11, color=SUBTLE)
    fig.savefig(FIGURES / "currents.png", facecolor=BG, pad_inches=0.2)
    plt.close(fig)


def lag_map():
    cells = pd.read_csv(RESULTS / f"lag_cells{SHOW}.csv")
    fig = plt.figure(figsize=(14, 9.5), dpi=250, facecolor=BG)
    ax = map_axes(fig, [0.02, 0.05, 0.9, 0.8], extent=(-50, 115, -50, 58), central=30)
    lim = np.nanpercentile(cells.loss_h_per_day, 95)
    cmap = LinearSegmentedColormap.from_list("lag", ["#24324a", "#3fb5a3", "#e8d36b", "#f28a3a", "#ff5a4a"])
    sc = ax.scatter(cells.cell_lon, cells.cell_lat, c=cells.loss_h_per_day.clip(lower=0),
                    s=np.clip(cells.n / 4, 12, 160), cmap=cmap, vmin=0, vmax=lim, transform=PC, zorder=5, lw=0,
                    alpha=0.92)
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.4])
    cb = fig.colorbar(sc, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("hours lost per day of sailing", color=SUBTLE)
    fig.text(0.03, 0.94, "Where the hours went", fontsize=24, fontweight="bold")
    fig.text(0.03, 0.905, "Hours per day by which the logged ships fell further behind the hindsight frontier, in each "
             "5-degree square (all voyages).  Dot size: ship-days.", fontsize=11, color=SUBTLE)
    fig.savefig(FIGURES / "lag_map.png", facecolor=BG, pad_inches=0.2)
    plt.close(fig)


def checks_fig(checks):
    from polar import Polar
    seg = pd.read_csv(DERIVED / "segments.csv")
    fig = plt.figure(figsize=(15, 5), dpi=220, facecolor=BG)
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    ax.set_facecolor(BG)
    pol = Polar.load()
    hull = Polar.load(DERIVED / "polar_hull.json")
    ang = np.linspace(0, 180, 181)
    for ws, c in ((3, "#3fb5a3"), (9, "#f2c14e")):
        r = pol(np.full_like(ang, ws), ang)
        ax.plot(np.radians(ang), r, color=c, lw=1.8, label=f"{ws} m/s, fitted")
        ax.plot(-np.radians(ang), r, color=c, lw=1.8)
        h = hull(np.full_like(ang, ws), ang)
        ax.plot(np.radians(ang), h, color=c, lw=1.2, ls="--", label=f"{ws} m/s, no-go zone")
        ax.plot(-np.radians(ang), h, color=c, lw=1.2, ls="--")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(colors=SUBTLE, labelsize=7)
    ax.grid(color=COAST)
    ax.set_title("Fleet polar, knots made good (wind from the top)", color=TEXT, fontsize=10, pad=18)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="lower right", bbox_to_anchor=(1.25, -0.05))
    ax = fig.add_subplot(1, 3, 2)
    dark_ax(ax)
    k4 = checks["K4"]
    r = seg.dropna(subset=["log_wdir"])
    err = np.abs((r.log_wdir - r.wind_from + 180) % 360 - 180)
    ax.hist(err, bins=np.arange(0, 181, 7.5), color=AMBER, alpha=0.85)
    ax.axvline(k4["median_err_real"], color=TEXT, lw=1.5)
    ax.axvline(k4["median_err_placebo"], color=CYAN, lw=1.5, ls="--")
    ax.set_title(f"K4: logged vs reanalysis wind direction\nmedian error {k4['median_err_real']:.0f} deg "
                 f"(wrong-year winds {k4['median_err_placebo']:.0f})", color=TEXT, fontsize=10, loc="left")
    ax.set_xlabel("angle between logged and 20CR wind (deg)")
    ax = fig.add_subplot(1, 3, 3)
    dark_ax(ax)
    k5 = checks["K5"]
    ax.bar([0, 1], [k5["median_rho_real"], k5["median_rho_placebo"]], color=[AMBER, CYAN], width=0.6)
    ax.set_xticks([0, 1], ["winds of the day", "wrong-year winds"])
    ax.axhline(0, color=SUBTLE, lw=0.8)
    ax.set_title(f"K5: does the day's wind predict the day's run?" + chr(10) + "median per-voyage rho; real beats "
                 f"placebo in {k5['frac_legs_real_better'] * 100:.0f}%", color=TEXT, fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "checks.png", facecolor=BG)
    plt.close(fig)


def main():
    res, paths, summary, checks = load(SHOW)
    leg_id = hero(res, paths)
    print("hero leg", leg_id)
    fleet(res, paths)
    rediscovery(res, paths, checks)
    ratios(summary)
    equator()
    if (RESULTS / "legs_routed_hull_cur_s42.csv").exists():
        currents_fig(summary)
    lag_map()
    checks_fig(checks)


if __name__ == "__main__":
    main()
