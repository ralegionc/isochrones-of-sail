# Isochrones of Sail: pre-registered plan

Written 2026-10-15, before any routing results were computed. Wind data had been
downloaded and the CLIWOC logbooks profiled (counts only), nothing else.

## Question

How much time did sailing captains of the 1836-1855 period leave on the table,
given the winds they actually met? And was the loss down to route choice, or
down to the weather they could not have known about?

## Data

* CLIWOC 2.1 logbooks (Steven Ottens' TSV release). After 1836 these are almost
  all Dutch, most of them on the Netherlands to Batavia run. One noon position
  per day, logged wind direction and force.
* NOAA-CIRES-DOE 20th Century Reanalysis v3, 10 m wind, ensemble mean (6-hourly,
  about 1.4 degree subset) and ensemble spread (daily). 20CRv3 assimilates surface
  pressure only, so the logged winds are independent of it.
* Natural Earth 1:50m land polygons for the coastline.

## Units of analysis

A leg is a run of consecutive daily positions at sea, at least 15 days long, with
no anchored days and no gap over 3 days. Leg ends are trimmed until they are at
least 30 km from land. Legs must lie inside the domain 80W-130E, 62S-66N.

## Model

* Ship polar: daily speed made good as a function of 20CR wind speed and the angle
  between the course made good and the wind, fitted as binned medians over all
  ship-days. One polar for the whole fleet.
* Four times per leg:
  * T_act: logged elapsed time.
  * T_rep: the logged track replayed through the 20CR winds with the polar.
  * T_opt: hindsight optimum, time-dependent shortest path through the real 20CR
    winds (0.25 degree grid, 32 headings).
  * T_clim: the route a planner with perfect climatology and no forecasts would
    choose (optimum through 1836-1855 month-wise expected speeds), then sailed
    through the real winds.
* R_hind = T_rep / T_opt and R_clim = T_rep / T_clim. Using T_rep rather than T_act
  in both ratios means the ship and the winds are modelled the same way on both
  sides, so the ratio measures route choice only.

## Known-answer checks (must pass before the hypotheses are read)

* K1. Uniform wind: the arrival-time contours from a point match the convex hull
  of the polar scaled by time, radial error under 3%.
* K2. Uniform wind: time to a target dead upwind and dead downwind within 3% of the
  analytic value (distance over best velocity made good).
* K3. Rediscovery: the climatology-optimal route from the Channel approaches to
  Java Head must cross the equator between 15W and 35W (the Portuguese volta do
  mar, keeping off the African coast) and cross 60E south of 30S (Brouwer's route
  through the westerlies), for at least 10 of the 12 departure months.
* K4. Logged wind directions agree better with 20CR on the same day than with
  20CR on the same calendar day in other years (paired test, p < 0.01).
* K5. Day-by-day replay speed correlates with the logged daily distance, and more
  strongly than with wrong-year winds (paired test, p < 0.01).
* K6. The fitted polar must be slower close-hauled (angle under 60 degrees) than on
  a broad reach, and faster in 8 m/s than in 3 m/s of wind.

Decision rule: if K4 or K5 fails, 20CR holds no day-specific weather along these
tracks. T_opt then reduces to a climatology-like bound and H1 is reported as not
interpretable.

## Hypotheses

* H1 (hindsight gap). Median R_hind across legs lies between 1.10 and 1.30: the
  hindsight route is 10-30% faster than the route sailed.
* H2 (captains versus climatology). Median R_clim is at most 1.00: captains did at
  least as well as perfect climatology without forecasts, because reading the
  local weather made up for imperfect charts. Test: bootstrap over ships, 95% CI.
* H3 (learning). R_clim falls with year over 1836-1855 (Spearman rho < 0,
  p < 0.05), with legs after 1850 closer to the climatological route than legs
  before 1845. Maury's wind and current charts appeared from 1848.

Anything not listed here is exploratory and will be labelled that way.

## Addendum 1 (2026-10-15, before the full run)

A smoke test on 1846-1847 (63 legs for the polar, 3 outbound legs routed) showed
that replaying a logged track takes 8-20% longer than the log says, because the
time along a track averages 1/speed. One global factor is now applied to the polar
so that the median replayed leg takes exactly its logged time. A single scale on
every speed leaves R_hind and R_clim almost unchanged. The same smoke test showed
the binned polar is flat in light winds (wind errors blur the angle). That is
left as is and not tuned; K3 decides whether the model is usable.

## Addendum 2 (2026-10-15, before the full run)

Added one exploratory robustness run, decided before any full-run result was seen:
the same routing with a polar fitted to the sailors' own logged wind force and
direction (after removing the fleet-wide mean offset between logged and reanalysis
directions). Routing still uses the reanalysis winds. Reason: reanalysis errors
blur the main polar, which makes headwinds look cheaper than they were and could
make the captains' long detours look wasteful. The main analysis is unchanged;
this run is reported as exploratory.

## Addendum 3 (2026-10-15, after K3 failed, before any routing of the voyages)

K3 failed as pre-registered: 7 of 12 months. From March to October the chart route
crosses the equator at 13.8-15.0W, just east of the 15W bar. The routes do swing west
of Africa, but only to 17-25W in the South Atlantic (logged outbound tracks reach a
median 30.6W), and they run their easting at 50-54S (logged tracks cross 60E at a
median 39.1S; 99% of all clean legs never go south of 42.0S). The model knows nothing
about ice, cold or heavy seas, and its polar makes headwinds look cheap, so it cuts
both corners.

Changes, decided before any voyage was routed:

1. A southern limit: routes may not go south of 42S, the fleet's own envelope (the
   1st percentile of each leg's southernmost latitude). Replays of logged tracks are
   not limited.
2. K3 is rerun for four models: {reanalysis-space polar, logged-wind polar} x
   {no limit, 42S limit}. All four are reported.
3. Model used to read H1-H3, in this order: the first of (a) reanalysis polar with the
   42S limit, (b) logged-wind polar with the 42S limit, that passes K3. If neither
   passes, H1-H3 are read from (a) and flagged as resting on a model that failed its
   rediscovery check. Results from every routed model are reported.

## Addendum 4 (2026-10-15, after all four K3 variants failed, before any voyage was routed)

All four models failed K3 (7, 3, 4 and 3 of 12 months). Logged outbound ships
crossed the equator at a steady median 23.0W in every departure month; the model
crosses at 19.5-21.8W for November-February departures but 13.8-15.2W for
March-October. Both fitted polars give 3.4-3.9 kn of progress within 60 degrees of
the wind, which a square-rigger cannot do directly: errors in the wind direction
(median 34 degrees against the logs) blur the angle.

One more exploratory model: the "tacking-hull" polar. It keeps the reanalysis-space
polar from 67.5 degrees off the wind outwards (six points, the usual limit for a
square-rigger) and replaces the closer sector with what tacking at 67.5 degrees can
make good: v(67.5) cos(67.5) / cos(angle). It gets the 42S limit, its own global
calibration and climatology, and its own K3 run. It is reported as exploratory
whatever K3 says, and does not replace the Addendum 3 primary model.

## Addendum 5 (2026-10-15, numerical fix after the first routing run)

In the first run the search timed each grid edge from the wind at two points, while
the replay integrates in 20 km steps. The search exploited that gap: its chosen
paths replayed 1-3% slower than it predicted (up to 60% on single voyages with the
tacking-hull polar), so in 26-39% of voyages the wind-chart route replayed faster
than the "hindsight optimum". The search now times each edge with the same 20 km
sub-steps as the replay, so search and replay agree on every path. As a definition,
the hindsight optimum is also never allowed to be slower than the wind-chart route
(if the chart route replays faster, it is the optimum). The first-run medians, for
the record: reanalysis polar with the 42S limit R_hind 1.154, R_clim 1.143;
tacking-hull polar R_hind 1.165, R_clim 1.159. All four models are rerun.

## Addendum 6 (2026-10-15, exploratory: ocean currents, after K3 failed)

The model so far ignores ocean currents, the most likely missing piece behind the K3
failure and behind the homeward ghosts beating up the African coast. This is an
exploratory model change made after K3 failed; it does not replace the Addendum 3
primary model or change any verdict.

Change: a monthly near-surface current climatology (NOAA Global Drifter Program,
Laurindo et al. 2017, v3.10, 0.25 degree) is added to the router and the replay. Over
the ground, a ship steering to hold a course makes good sqrt(v^2 - c_cross^2) + c_along,
where v is the polar speed through the water and c_cross, c_along are the current
components across and along the course (0.3 kn floor as before). The polar is refitted
on speed through the water (the daily ground run minus the mean current along it),
then recalibrated. Two models get currents, both with the 42S limit: the reanalysis
polar and the tacking-hull polar.

Reported whatever the outcome: (a) where the chart routes cross the equator against the
logged 23W, (b) whether the reanalysis polar now passes K3, (c) whether the homeward
ghosts move west around the Azores (longitude where routes cross 30N, logged against
ghost), and (d) how R_clim changes.
