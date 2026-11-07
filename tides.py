"""Tidal current / height prediction with pyTMD."""
import json
import os
import urllib.parse
import urllib.request
from functools import lru_cache
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

# Sources: label -> pyTMD model names to try (first one installed is used).
# Only models containing both elevation and current (u/v) constituents.
SOURCES = {
    "TPXO9": ["TPXO9-atlas-v5", "TPXO9-atlas-v4", "TPXO10-atlas-v2"],
    "FES": ["FES2014"],
}
NOAA = "NOAA CO-OPS API"
NOAA_MAX_CURRENT_KM = 30.0
NOAA_MAX_TIDE_KM = 100.0
KN_PER_CMS = 0.0194384  # cm/s -> knots
# touch-friendly interactive charts: pinch to zoom, drag to pan, double-tap to reset
# NB: Plotly only zooms on a two-finger pinch in dragmode "zoom" (in "pan" the pinch just pans).
# Panning is then done with the modebar's Pan button, which is always shown (no hover on touch screens).
PLOT_CONFIG = dict(displaylogo=False, scrollZoom=True, doubleClick="reset", displayModeBar=True,
                   modeBarButtonsToRemove=["select2d", "lasso2d"], toImageButtonOptions=dict(scale=2))


def model_directory():
    """Directory holding the tide model files (override with PYTMD_DIR)."""
    return Path(os.environ.get("PYTMD_DIR", Path.home() / ".cache" / "pytmd"))


class NoDataError(Exception):
    """The selected source has no data at or near the requested point."""


def _installed_model(label):
    """Name of the first installed pyTMD model for a source label, or None."""
    import pyTMD.io
    for name in SOURCES[label]:
        try:
            pyTMD.io.model(model_directory()).from_database(name)
            return name
        except Exception:
            continue
    return None


def available_sources():
    """Sources that can be selected: installed models, plus NOAA (always)."""
    return [l for l in SOURCES if _installed_model(l)] + [NOAA]


def _as_array(res, key):
    """Extract a variable from a pyTMD result (DataTree, Dataset or DataArray)."""
    try:
        res = res[key]
    except (KeyError, TypeError, IndexError):
        pass
    if hasattr(res, "to_dataarray"):
        res = res.to_dataarray()
    return np.asarray(getattr(res, "values", res), dtype=float).ravel()


def _time_axis(center_date, days, step_min):
    start = np.datetime64(datetime(center_date.year, center_date.month, center_date.day)
                          - timedelta(days=days))
    n = int((2 * days + 1) * 24 * 60 / step_min)
    return start + np.arange(n) * np.timedelta64(step_min, "m")


def predict_tmd(lat, lon, center_date, model, days=3, step_min=10):
    """Predict currents and tide height for center_date +/- days (UTC).

    Returns a dict with times (datetime64), u/v (cm/s, east/north), height (m).
    """
    import pyTMD.compute

    directory = model_directory()
    times = _time_axis(center_date, days, step_min)
    n = len(times)
    x = np.full(n, lon, dtype=float)
    y = np.full(n, lat, dtype=float)
    kw = dict(directory=directory, model=model, type="time series",
              standard="datetime", extrapolate=True, cutoff=50.0)
    cur = pyTMD.compute.tide_currents(x, y, times, **kw)
    ele = pyTMD.compute.tide_elevations(x, y, times, **kw)
    u, v = _as_array(cur, "u"), _as_array(cur, "v")
    h = _as_array(ele, "z")
    if np.all(np.isnan(u)) or np.all(np.isnan(v)):
        raise NoDataError("The model has no tidal current data at or near this location.")
    return dict(times=times, u=u, v=v, height=h)


def analyse_currents(times, u, v, height):
    """Principal flow axis, signed speed (+flood / -ebb) and extrema."""
    ok = ~(np.isnan(u) | np.isnan(v))
    uu, vv = np.where(ok, u, 0.0), np.where(ok, v, 0.0)
    # principal axis of the velocity ellipse
    theta = 0.5 * np.arctan2(2 * np.sum(uu * vv), np.sum(uu**2) - np.sum(vv**2))
    eu, ev = np.cos(theta), np.sin(theta)  # unit vector (east, north)
    speed = uu * eu + vv * ev
    # flood = water flowing while the tide rises
    dh = np.gradient(np.nan_to_num(height))
    if np.sum(speed * dh) < 0:
        eu, ev, speed = -eu, -ev, -speed
    flood_bearing = float(np.degrees(np.arctan2(eu, ev)) % 360)  # direction flowing TO
    ebb_bearing = (flood_bearing + 180) % 360
    return dict(speed=speed, flood_bearing=flood_bearing, ebb_bearing=ebb_bearing,
                extrema=find_extrema(speed))


def find_zeros(speed):
    """Sample positions (fractional index) where the current is zero (slack)."""
    out = []
    n = len(speed)
    i = 0
    while i < n:
        if speed[i] == 0:
            out.append(float(i))
            while i + 1 < n and speed[i + 1] == 0:  # skip a run of zeros
                i += 1
        elif i + 1 < n and speed[i] * speed[i + 1] < 0:
            out.append(i + speed[i] / (speed[i] - speed[i + 1]))
        i += 1
    return out


def find_extrema(speed):
    """Indices of local maxima of positive flow and local minima of negative flow.

    Flat tops (equal consecutive samples) are handled; the middle sample is used.
    """
    n = len(speed)
    thr = 0.02 * np.max(np.abs(speed))
    out = []
    i = 1
    while i < n - 1:
        j = i
        while j + 1 < n and speed[j + 1] == speed[i]:
            j += 1
        if j + 1 >= n:
            break
        v = speed[i]
        if abs(v) > thr and ((v > 0 and speed[i - 1] < v > speed[j + 1]) or
                             (v < 0 and speed[i - 1] > v < speed[j + 1])):
            out.append((i + j) // 2)
        i = j + 1
    return np.array(out, dtype=int)


def _get_json(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


@lru_cache(maxsize=4)
def _noaa_stations(kind):
    d = _get_json("https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json",
                  {"type": kind})
    return [(s["id"], s["name"], s["lat"], s["lng"]) for s in d["stations"]
            if s.get("lat") is not None and s.get("lng") is not None]


def _nearest_list(kind, lat, lon, max_km, n=5):
    """Up to n stations within max_km, nearest first, as (km, id, name)."""
    out = []
    for sid, name, slat, slon in _noaa_stations(kind):
        # haversine distance
        p1, p2 = np.radians(lat), np.radians(slat)
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(slon - lon) / 2) ** 2
        km = 12742 * np.arcsin(np.sqrt(a))
        if km <= max_km:
            out.append((float(km), sid, name))
    return sorted(out)[:n]


def _nearest(kind, lat, lon, max_km):
    found = _nearest_list(kind, lat, lon, max_km, 1)
    return found[0] if found else None


def _noaa_data(product, station, center_date, days, extra):
    b = center_date - timedelta(days=days)
    e = center_date + timedelta(days=days)
    p = dict(product=product, station=station, begin_date=b.strftime("%Y%m%d"),
             end_date=e.strftime("%Y%m%d"), time_zone="gmt", units="metric",
             format="json", application="pymeteo")
    p.update(extra)
    d = _get_json("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter", p)
    if "error" in d:
        raise NoDataError(d["error"].get("message", "NOAA returned no data").strip())
    return d


def _noaa_times(rows):
    return np.array([np.datetime64(r["Time"] if "Time" in r else r["t"]) for r in rows])


def _noaa_current_series(sid, name, center_date, days):
    """Current predictions (times, speed cm/s +flood, flood bearing, ebb bearing) of one station."""
    d = _noaa_data("currents_predictions", sid, center_date, days, {"interval": 6})
    rows = d["current_predictions"]["cp"]
    if not rows:
        raise NoDataError(f"NOAA station {name} returned no predictions.")
    times = _noaa_times(rows)
    # NOAA Velocity_Major is in cm/s for metric units, positive = flood
    speed = np.array([r["Velocity_Major"] for r in rows], dtype=float)
    if any("Type" in r for r in rows):
        # subordinate station: only slack/max events -> smooth cosine between them
        t0 = times.astype("datetime64[m]").astype("int64")
        # regular 10-min grid plus the event times, so published peaks are exact
        grid = np.union1d(np.arange(t0[0], t0[-1] + 1, 10), t0)
        k = np.clip(np.searchsorted(t0, grid, side="right") - 1, 0, len(t0) - 2)
        f = (grid - t0[k]) / (t0[k + 1] - t0[k])
        w = (1 - np.cos(np.pi * f)) / 2
        speed = speed[k] * (1 - w) + speed[k + 1] * w
        times = grid.astype("datetime64[m]")
    flood = float(rows[0]["meanFloodDir"])
    ebb = float(rows[0]["meanEbbDir"])
    return times, speed, flood, ebb


def predict_noaa(lat, lon, center_date, days=3):
    """Tide predictions from the nearest NOAA CO-OPS current / tide stations."""
    cs = _nearest("currentpredictions", lat, lon, NOAA_MAX_CURRENT_KM)
    if cs is None:
        raise NoDataError(f"No NOAA CO-OPS current prediction station within {NOAA_MAX_CURRENT_KM:.0f} km.")
    km, sid, name = cs
    times, speed, flood, ebb = _noaa_current_series(sid, name, center_date, days)
    height, ts = None, None
    # some tide stations lack a given datum or have no data: try the next datum/station
    for cand in _nearest_list("tidepredictions", lat, lon, NOAA_MAX_TIDE_KM):
        for datum, interval in [(d, i) for i in (6, "hilo") for d in ("MLLW", "MSL", "MTL", "STND")]:
            try:
                td = _noaa_data("predictions", cand[1], center_date, days,
                                {"interval": interval, "datum": datum})
                tt = _noaa_times(td["predictions"])
                hv = np.array([float(r["v"]) for r in td["predictions"]])
                if interval == "hilo":  # only high/low events: cosine between them
                    t0, tq = tt.astype("datetime64[m]").astype("int64"), times.astype("datetime64[m]").astype("int64")
                    k = np.clip(np.searchsorted(t0, tq, side="right") - 1, 0, len(t0) - 2)
                    f = np.clip((tq - t0[k]) / (t0[k + 1] - t0[k]), 0, 1)
                    w = (1 - np.cos(np.pi * f)) / 2
                    height = hv[k] * (1 - w) + hv[k + 1] * w
                else:
                    height = np.interp(times.astype("int64"), tt.astype("int64"), hv)
                ts = cand
                break
            except Exception:
                continue
        if height is not None:
            break
    return dict(times=times, speed=speed, flood_bearing=flood, ebb_bearing=ebb, height=height,
                source=f"NOAA CO-OPS station {name} ({sid}), {km:.1f} km away"
                       + (f"; tide height from {ts[2]} ({ts[0]:.0f} km)" if height is not None else ""))


def local_timezone(lat, lon):
    """ZoneInfo for the coordinate (UTC if it cannot be determined)."""
    try:
        from timezonefinder import TimezoneFinder
        name = TimezoneFinder().timezone_at(lat=lat, lng=lon)
        return ZoneInfo(name) if name else ZoneInfo("UTC")
    except Exception:
        return ZoneInfo("UTC")


def get_tides(source, lat, lon, center_date, use_local=True):
    """Return dict(times, speed (cm/s, +flood), flood_bearing, ebb_bearing, height|None, extrema, source)."""
    if source == NOAA:
        r = predict_noaa(lat, lon, center_date, days=4)
    else:
        model = _installed_model(source)
        if model is None:
            raise NoDataError(f"{source} model files are not installed.")
        t = predict_tmd(lat, lon, center_date, model, days=4)
        a = analyse_currents(t["times"], t["u"], t["v"], t["height"])
        r = dict(times=t["times"], speed=a["speed"], flood_bearing=a["flood_bearing"],
                 ebb_bearing=a["ebb_bearing"], height=t["height"], extrema=a["extrema"],
                 source=f"{model} model")
    # crop to center_date +/- 3 days, in local calendar days or in UTC
    tz = local_timezone(lat, lon) if use_local else ZoneInfo("UTC")
    first = datetime(center_date.year, center_date.month, center_date.day) - timedelta(days=3)
    start = first.replace(tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    end = (first + timedelta(days=7)).replace(tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    keep = (r["times"] >= np.datetime64(start)) & (r["times"] < np.datetime64(end))
    if not keep.any():
        raise NoDataError("No predictions for the selected period.")
    r["times"], r["speed"] = r["times"][keep], r["speed"][keep]
    if r["height"] is not None:
        r["height"] = r["height"][keep]
    r["extrema"] = find_extrema(r["speed"])
    # shift to wall-clock time of the chosen zone (handles DST changes)
    utc = r["times"].astype("datetime64[s]").astype(object)
    local = [d.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz) for d in utc]
    r["times"] = np.array([np.datetime64(d.replace(tzinfo=None)) for d in local])
    r["tzlabel"] = f"local time, {tz.key}" if use_local else "UTC"
    return r


def predict_tmd_points(lats, lons, times, model):
    """East/north current (cm/s) at paired points/times (UTC datetime64) from a pyTMD model."""
    import pyTMD.compute
    kw = dict(directory=model_directory(), model=model, type="drift",
              standard="datetime", extrapolate=True, cutoff=50.0)
    cur = pyTMD.compute.tide_currents(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float),
                                      np.asarray(times), **kw)
    return _as_array(cur, "u"), _as_array(cur, "v")


@lru_cache(maxsize=64)
def _noaa_station_series(sid, name, day):
    try:
        return _noaa_current_series(sid, name, day, 1)
    except Exception:
        return None  # cached too: a station without data is not asked again


@lru_cache(maxsize=2048)
def _nearest_current_station(lat, lon):
    return _nearest("currentpredictions", lat, lon, NOAA_MAX_CURRENT_KM)  # local, station list is cached


def _noaa_route_current(lat, lon, t):
    """Current (cm/s, signed +flood) and flood/ebb bearings at one point/time; one request per station and day."""
    cs = _nearest_current_station(round(lat, 4), round(lon, 4))
    if cs is None:
        raise NoDataError("No NOAA station nearby.")
    _, sid, name = cs
    series = _noaa_station_series(sid, name, t.astype(object).date())
    if series is None:
        raise NoDataError(f"NOAA station {name} has no data.")
    times, speed, flood, ebb = series
    ts = times.astype("datetime64[s]").astype("int64")
    return float(np.interp(t.astype("int64"), ts, speed)), flood, ebb


def current_vectors(source, lats, lons, times):
    """Tidal current (east_kn, north_kn) at each (lat, lon, UTC datetime64 time).

    Entries are None where the source has no data.
    """
    times = np.asarray(times, dtype="datetime64[s]")
    if source == NOAA:
        out = []
        for lat, lon, t in zip(lats, lons, times):
            try:
                sp, flood, ebb = _noaa_route_current(float(lat), float(lon), t)
                sp *= KN_PER_CMS
                brg = np.radians(flood if sp >= 0 else ebb)
                out.append((abs(sp) * np.sin(brg), abs(sp) * np.cos(brg)))
            except Exception:
                out.append(None)
        return out
    model = _installed_model(source)
    if model is None:
        raise NoDataError(f"{source} model files are not installed.")
    u, v = predict_tmd_points(lats, lons, times, model)
    return [None if (np.isnan(a) or np.isnan(b)) else (a * KN_PER_CMS, b * KN_PER_CMS)
            for a, b in zip(u, v)]


def plot_compass(flood_bearing, ebb_bearing):
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    for bearing, color in ((flood_bearing, "red"), (ebb_bearing, "blue")):
        ax.annotate("", xy=(np.radians(bearing), 0.9), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=3, mutation_scale=20))
    ax.set_title(f"Flood {flood_bearing:.0f}° (red)\nEbb {ebb_bearing:.0f}° (blue)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_tides(times, speed, height, extrema, tzlabel="UTC", height_unit="m"):
    t = times.astype("datetime64[s]").astype(object)
    x = mdates.date2num(t)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if height is not None:
        ax2 = ax.twinx()
        h = np.asarray(height) * (3.28084 if height_unit == "ft" else 1.0)
        ax2.plot(t, h, color="gray", lw=1.2, alpha=0.8)
        ax2.set_ylabel(f"Tide height ({height_unit})", color="gray")
    kn = speed * KN_PER_CMS
    # insert zero crossings so each colour change happens exactly at zero
    xs, ys = [x[0]], [kn[0]]
    for i in range(1, len(x)):
        if kn[i - 1] * kn[i] < 0:
            f = kn[i - 1] / (kn[i - 1] - kn[i])
            xs.append(x[i - 1] + f * (x[i] - x[i - 1]))
            ys.append(0.0)
        xs.append(x[i])
        ys.append(kn[i])
    xs, ys = np.array(xs), np.array(ys)
    pts = np.column_stack([xs, ys])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    mid = (ys[:-1] + ys[1:]) / 2
    ax.add_collection(LineCollection(segs, colors=["red" if m >= 0 else "blue" for m in mid],
                                     linewidths=1.8))
    ax.set_xlim(x[0], x[-1])
    pad = 0.25 * (ys.max() - ys.min())
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_ylabel("Current speed (kn)  [red + flood / blue − ebb]")
    for i in extrema:
        c = "red" if kn[i] >= 0 else "blue"
        ax.plot(t[i], kn[i], "o", color=c, ms=4)
        ax.annotate(f"{abs(kn[i]):.1f} kn\n{t[i]:%H:%M}", (t[i], kn[i]), fontsize=7,
                    ha="center", va="bottom" if kn[i] >= 0 else "top",
                    xytext=(0, 5 if kn[i] >= 0 else -5), textcoords="offset points", color=c)
    secs = times.astype("datetime64[s]").astype("int64")
    for z in find_zeros(speed):
        i0 = int(np.floor(z))
        i1 = min(i0 + 1, len(secs) - 1)
        tz_ = datetime.utcfromtimestamp(secs[i0] + (z - i0) * (secs[i1] - secs[i0]))
        ax.plot(tz_, 0, "|", color="black", ms=10, mew=1.5)
        ax.annotate(tz_.strftime("%H:%M"), (tz_, 0), fontsize=6.5, rotation=90, color="dimgray",
                    ha="center", va="top", xytext=(0, -7), textcoords="offset points")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.grid(alpha=0.3)
    ax.set_title(f"Tidal current and tide height — times {tzlabel}\n(black ticks: zero current)", fontsize=10)
    fig.tight_layout()
    return fig


def compass_plotly(flood_bearing, ebb_bearing):
    """Interactive compass with the flood (red) and ebb (blue) directions."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, brg, color in ((f"Flood {flood_bearing:.0f}°", flood_bearing, "red"),
                             (f"Ebb {ebb_bearing:.0f}°", ebb_bearing, "blue")):
        fig.add_trace(go.Scatterpolar(
            theta=[brg, brg], r=[0, 1], mode="lines+markers", name=name, line=dict(color=color, width=4),
            marker=dict(symbol="arrow", size=[0, 18], angleref="previous", color=color)))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1.05]),
                   angularaxis=dict(rotation=90, direction="clockwise", tickmode="array",
                                    tickvals=[0, 90, 180, 270], ticktext=["N", "E", "S", "W"])),
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=20, b=20), height=320, dragmode="pan")
    return fig


def tides_plotly(times, speed, height, extrema, tzlabel="UTC", height_unit="m"):
    """Interactive tidal current (red flood / blue ebb) and tide height time series."""
    import plotly.graph_objects as go
    secs = times.astype("datetime64[s]").astype("int64").astype(float)
    kn = np.asarray(speed, dtype=float) * KN_PER_CMS
    xs, ys = [secs[0]], [kn[0]]
    for i in range(1, len(secs)):  # insert zero crossings so colours change exactly at zero
        if kn[i - 1] * kn[i] < 0:
            f = kn[i - 1] / (kn[i - 1] - kn[i])
            xs.append(secs[i - 1] + f * (secs[i] - secs[i - 1]))
            ys.append(0.0)
        xs.append(secs[i])
        ys.append(kn[i])
    xs, ys = np.array(xs), np.array(ys)
    dt = lambda a: np.asarray(a).astype("int64").astype("datetime64[s]")
    fig = go.Figure()
    if height is not None:
        h = np.asarray(height) * (3.28084 if height_unit == "ft" else 1.0)
        fig.add_trace(go.Scatter(x=dt(secs), y=h, name=f"Tide height ({height_unit})", yaxis="y2",
                                 line=dict(color="gray", width=1.2), opacity=0.8,
                                 hovertemplate=f"%{{y:.2f}} {height_unit}<extra>height</extra>"))
    fig.add_trace(go.Scatter(x=dt(xs), y=np.where(ys >= 0, ys, np.nan), name="Flood", line=dict(color="red", width=2),
                             hovertemplate="%{y:.2f} kn<extra>flood</extra>"))
    fig.add_trace(go.Scatter(x=dt(xs), y=np.where(ys <= 0, ys, np.nan), name="Ebb", line=dict(color="blue", width=2),
                             hovertemplate="%{y:.2f} kn<extra>ebb</extra>"))
    ex = np.asarray(extrema, dtype=int)
    if len(ex):
        fig.add_trace(go.Scatter(
            x=dt(secs[ex]), y=kn[ex], mode="markers+text", showlegend=False,
            customdata=[f"{abs(kn[i]):.2f} kn at {dt(secs[i]).astype(object):%H:%M}" for i in ex],
            hovertemplate="%{customdata}<extra></extra>",
            marker=dict(color=["red" if kn[i] >= 0 else "blue" for i in ex], size=6),
            text=[f"{abs(kn[i]):.1f} kn<br>{dt(secs[i]).astype(object):%H:%M}" for i in ex],
            textposition=["top center" if kn[i] >= 0 else "bottom center" for i in ex],
            textfont=dict(size=12, color=["red" if kn[i] >= 0 else "blue" for i in ex])))
    zs = find_zeros(speed)
    if zs:
        zt = [secs[int(np.floor(z))] + (z - np.floor(z)) * (secs[min(int(np.floor(z)) + 1, len(secs) - 1)] - secs[int(np.floor(z))])
              for z in zs]
        fig.add_trace(go.Scatter(x=dt(zt), y=[0] * len(zt), mode="markers+text", name="Zero current (slack)",
                                 marker=dict(symbol="line-ns-open", color="black", size=10, line=dict(width=1.5)),
                                 hovertemplate="slack %{x|%H:%M}<extra></extra>"))
        for z in zt:  # scatter text cannot be rotated: use annotations, vertical and below the zero line
            fig.add_annotation(x=dt(z), y=0, text=f"{dt(z).astype(object):%H:%M}", textangle=-90,
                               showarrow=False, yanchor="top", yshift=-8, font=dict(size=12, color="black"))
    pad = 0.3 * (ys.max() - ys.min() or 1)
    fig.update_layout(
        title=dict(text=f"Tidal current and tide height — {tzlabel}", x=0.01, xanchor="left", font=dict(size=13)),
        xaxis=dict(gridcolor="rgba(128,128,128,0.25)", tickformatstops=[  # hours once zoomed in, days otherwise
            dict(dtickrange=[None, 86400000], value="%H:%M<br>%a %d"),
            dict(dtickrange=[86400000, None], value="%a %d %b")]),
        yaxis=dict(title="Current (kn) + flood / − ebb", range=[ys.min() - pad, ys.max() + pad], fixedrange=True,
                   zeroline=True, gridcolor="rgba(128,128,128,0.25)"),
        yaxis2=dict(title=f"Height ({height_unit})", overlaying="y", side="right", showgrid=False, color="gray",
                    fixedrange=True),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=60, b=10), height=420, dragmode="zoom", hovermode="closest")
    return fig


def tide_figures_plotly(r, height_unit="m"):
    return (compass_plotly(r["flood_bearing"], r["ebb_bearing"]),
            tides_plotly(r["times"], r["speed"], r["height"], r["extrema"], r["tzlabel"], height_unit))


def tide_figures(r, height_unit="m"):
    """Compass and time-series figures from a get_tides() result."""
    return (plot_compass(r["flood_bearing"], r["ebb_bearing"]),
            plot_tides(r["times"], r["speed"], r["height"], r["extrema"], r["tzlabel"], height_unit))
