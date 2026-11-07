#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:51:29 2023

@author: francesco
"""

from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from meteostat import Point, hourly, interpolate, stations
import math
import json
import urllib.parse
import urllib.request
import numpy as np


coordinates = "43.351364, 10.443741"
month = 7
day = 2

average_years = 7

now = datetime.now()
current_year = now.year
desired_date = datetime(current_year, month, day)

coords = coordinates.split(',')
coords = [float(c.strip()) for c in coords]

METEOSTAT = "Meteostat"
OPENMETEO = "Open-Meteo"
WEATHER_SOURCES = [OPENMETEO, METEOSTAT]


def _openmeteo_year(lat, lon, start, end):
    """Hourly wind (km/h), direction, temperature and precipitation from Open-Meteo."""
    q = urllib.parse.urlencode(dict(
        latitude=lat, longitude=lon, start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d'), wind_speed_unit='kmh', timezone='GMT',
        hourly='wind_speed_10m,wind_direction_10m,temperature_2m,precipitation'))
    with urllib.request.urlopen('https://archive-api.open-meteo.com/v1/archive?' + q, timeout=60) as r:
        h = json.load(r)['hourly']
    nan = lambda v: math.nan if v is None else v
    return ([nan(v) for v in h['wind_speed_10m']], [nan(v) for v in h['wind_direction_10m']],
            [v for v in h['temperature_2m'] if v is not None],
            [v for v in h['precipitation'] if v is not None])


# (label, angle in the plot [deg, counter-clockwise from east], colour)
DIRECTIONS = [("N", 90, 'red'), ("NE", 45, 'orange'), ("E", 0, 'yellow'), ("SE", -45, 'greenyellow'),
              ("S", -90, 'green'), ("SW", -135, 'turquoise'), ("W", 180, 'blue'), ("NW", 135, 'blueviolet')]
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
          'October', 'November', 'December']


def wind_data(coords, desired_date, min_speed=5, source=OPENMETEO):
    """Typical wind statistics for the week around desired_date, at coords (lat, lon)."""
    location = Point(coords[0], coords[1], 0)
    nearby_stations = stations.nearby(location, radius=100000, limit=10) if source == METEOSTAT else None
    now = datetime.now()
    current_year = now.year

    if isinstance(desired_date, date):
        desired_date = datetime(desired_date.year, desired_date.month, desired_date.day)

    month = desired_date.month
    day = desired_date.day

    # if date is in the past, use current year for averaging
    if desired_date < now:
        starting_year = current_year
    else:
        starting_year = current_year - 1

    def get_data_for_year(year, month, day, point, station_ids):
        midpoint = datetime(year, month, day)
        start = midpoint - timedelta(days=3)
        end = midpoint + timedelta(days=3)
        ts = hourly(station_ids, start, end)
        data = interpolate(ts, point).fetch()
        return data

    wind_speeds = []
    wind_dirs = []
    temps = []
    precip = []

    for year in range(starting_year-average_years+1, starting_year+1):
        if source == OPENMETEO:
            midpoint = datetime(year, month, day)
            ws, wd, t, p = _openmeteo_year(coords[0], coords[1], midpoint - timedelta(days=3),
                                           midpoint + timedelta(days=3))
            wind_speeds.extend(ws)
            wind_dirs.extend(wd)
            temps.extend(t)
            precip.extend(p)
            continue
        d = get_data_for_year(year, month, day, location, nearby_stations)
        wind_speeds.extend(d['wspd'] if 'wspd' in d else [math.nan]*len(d))
        wind_dirs.extend(d['wdir'] if 'wdir' in d else [math.nan]*len(d))
        if 'temp' in d:
            temps.extend(d['temp'].dropna())
        if 'prcp' in d:
            precip.extend(d['prcp'].dropna())

    # sanitize wind, convert to knots
    pairs = [(s / 1.852, d) for s, d in zip(wind_speeds, wind_dirs) if not (math.isnan(s) or math.isnan(d))]
    total = len(pairs)

    def sector(d):
        """Index in DIRECTIONS of the 45-degree sector (centred on the compass point) containing d."""
        for i, (_, angle, _) in enumerate(DIRECTIONS):
            centre = (90 - angle) % 360  # compass bearing of the sector centre
            if centre == 0:
                if d >= 360 - 22.5 or d < 22.5:
                    return i
            elif centre - 22.5 <= d < centre + 22.5:
                return i
        return 0

    low = 0
    per_dir = [[] for _ in DIRECTIONS]
    for s, d in pairs:
        if s < min_speed:
            low += 1
        else:
            per_dir[sector(d)].append(s)

    fractions = [len(v) / total for v in per_dir]
    max_fraction = max(fractions)
    width_factor = 0.125 / max_fraction if max_fraction > 0.125 else 1.0
    medians = [float(np.median(v)) if v else 0 for v in per_dir]
    tops = [float(np.percentile(v, 99)) if v else 0 for v in per_dir]
    low_t, high_t = np.percentile(temps, 5), np.percentile(temps, 95)
    rain = float(len([p for p in precip if p > 0])) / len(precip) * 100
    return dict(
        min_speed=min_speed, fraction_low=low / total, fractions=fractions, width_factor=width_factor,
        medians=medians, tops=tops, low_temp=float(low_t), high_temp=float(high_t), rain=rain,
        title=f'Week of {MONTHS[desired_date.month-1]} {desired_date.day}, '
              f'Loc: {location.latitude:.2f}, {location.longitude:.2f} ({source})',
        footer=f'Low temp: {low_t:.1f}°C, High temp: {high_t:.1f}°C, Precipitations {rain:.1f}%, '
               f'(Cutoff windspeed: {min_speed} kn)')


def _wedge_points(angle, half_width, radius, n=24):
    """Outline (x, y) of a pie wedge centred on angle (deg), apex at the origin."""
    a = np.radians(np.linspace(angle - half_width, angle + half_width, n))
    return [0.0, *(radius * np.cos(a))], [0.0, *(radius * np.sin(a))]


def wind_figures(w):
    """Two interactive Plotly figures (median speeds, top speeds): pinch/zoom/pan work on touch screens."""
    import plotly.graph_objects as go
    figs = []
    for key, name in (("medians", "Median speeds"), ("tops", "Top speeds")):
        fig = go.Figure()
        values = w[key]
        for (label, angle, color), frac, radius in zip(DIRECTIONS, w["fractions"], values):
            if radius <= w["min_speed"]:
                continue
            x, y = _wedge_points(angle, 180 * frac * w["width_factor"], radius)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", fill="toself", fillcolor=color, line=dict(color="black", width=0.5),
                hoverinfo="text", showlegend=False,
                text=f"{label}: {frac * 100:.0f}% of the time<br>{name.split()[0].lower()} {radius:.1f} kn"))
            fig.add_annotation(x=radius * 1.2 * math.cos(math.radians(angle)),
                               y=radius * 1.2 * math.sin(math.radians(angle)),
                               text=f"{radius:.1f}", showarrow=False, font=dict(size=13))
        ang = np.linspace(0, 2 * np.pi, 91)  # full circle, no apex point (which drew a radius line)
        fig.add_trace(go.Scatter(x=w["min_speed"] * np.cos(ang), y=w["min_speed"] * np.sin(ang), mode="lines",
                                 fill="toself", fillcolor="azure", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=0, y=0, text=f"{w['fraction_low'] * 100:.0f}%", showarrow=False, font=dict(size=13))
        lim = max(max(values), w["min_speed"]) * 1.35
        fig.update_xaxes(range=[-lim, lim], visible=False, constrain="domain")
        fig.update_yaxes(range=[-lim, lim], visible=False, scaleanchor="x", scaleratio=1)
        fig.update_layout(title=dict(text=name, x=0.02, xanchor="left", font=dict(size=16)), margin=dict(l=5, r=5, t=50, b=5),
                          height=420, dragmode="pan", plot_bgcolor="white")
        figs.append(fig)
    return figs


def plot_winds(coords, desired_date, min_speed=5, source=OPENMETEO):
    """Matplotlib version of the wind plots (used by the Qt UI)."""
    w = wind_data(coords, desired_date, min_speed, source)
    f, axes = plt.subplots(1, 2)
    for ax, key, name in ((axes[0], "medians", "Median speeds"), (axes[1], "tops", "Top speeds")):
        values = w[key]
        for (label, angle, color), frac, radius in zip(DIRECTIONS, w["fractions"], values):
            if radius <= min_speed:
                continue
            half = 180 * frac * w["width_factor"]
            ax.add_patch(patches.Wedge((0, 0), radius, angle - half, angle + half, fill=True, facecolor=color))
            ax.text(math.cos(math.radians(angle)) * radius * 1.2, math.sin(math.radians(angle)) * radius * 1.2,
                    f'{radius:.1f}', ha='center', va='center', fontsize=10)
        ax.add_patch(patches.Circle((0, 0), radius=min_speed, fc='azure'))
        ax.text(0, 0, f"{w['fraction_low']*100:.0f}%", ha='center', va='center', fontsize=8)
        lim = max(values) * 1.25
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(name)
    plt.text(0.5, 0, w["footer"], ha='center', va='bottom', transform=f.transFigure)
    plt.text(0.5, 0.9, w["title"], ha='center', va='bottom', transform=f.transFigure)


if __name__ == "__main__":
    plot_winds(coords, desired_date)
    plt.show()
