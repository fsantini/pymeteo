"""PDF export for the Streamlit pages: OpenStreetMap basemap + matplotlib plots, no browser needed."""
import io
import math
import urllib.request
from functools import lru_cache

import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
R_EARTH = 6378137.0
ATTRIBUTION = "Map © OpenStreetMap contributors"
A4 = (8.27, 11.69)  # portrait, inches
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]


def merc(lat, lon):
    """Web Mercator metres (x, y)."""
    lat = max(min(lat, 85.05), -85.05)
    return (R_EARTH * math.radians(lon),
            R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def _unmerc(x, y):
    return (math.degrees(2 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2), math.degrees(x / R_EARTH))


def _tile_xy(x, y, z):
    """Fractional tile coordinates of a Mercator point."""
    half = math.pi * R_EARTH
    n = 2 ** z
    return (x + half) / (2 * half) * n, (half - y) / (2 * half) * n


@lru_cache(maxsize=512)
def _tile(z, x, y):
    req = urllib.request.Request(TILE_URL.format(z=z, x=x % 2 ** z, y=y),
                                 headers={"User-Agent": "pymeteo-pdf-export/1.0 (+https://github.com/fsantini/pymeteo)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception:
        return Image.new("RGB", (256, 256), (225, 225, 225))


def draw_basemap(ax, x0, y0, x1, y1, zoom):
    """Draw OSM tiles at the given zoom on ax (data coordinates = Mercator metres) and set the view."""
    half = math.pi * R_EARTH
    n = 2 ** zoom
    tx0, ty0 = _tile_xy(x0, y1, zoom)
    tx1, ty1 = _tile_xy(x1, y0, zoom)
    ix0, ix1 = int(math.floor(tx0)), int(math.floor(tx1))
    iy0, iy1 = max(int(math.floor(ty0)), 0), min(int(math.floor(ty1)), n - 1)
    canvas = Image.new("RGB", ((ix1 - ix0 + 1) * 256, (iy1 - iy0 + 1) * 256))
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            canvas.paste(_tile(zoom, ix, iy), ((ix - ix0) * 256, (iy - iy0) * 256))
    size = 2 * half / n
    ext = (-half + ix0 * size, -half + (ix1 + 1) * size, half - (iy1 + 1) * size, half - iy0 * size)
    ax.imshow(np.asarray(canvas), extent=ext, origin="upper", interpolation="none", zorder=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.995, 0.005, ATTRIBUTION, transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5), zorder=20)


def _fit_zoom(width_m, height_m, w_px, h_px, max_zoom=16):
    """Largest zoom whose tile size fits the box in about w_px x h_px pixels (OSM tiles are 256 px)."""
    for z in range(max_zoom, 0, -1):
        px_per_m = 256 * 2 ** z / (2 * math.pi * R_EARTH)
        if width_m * px_per_m <= w_px and height_m * px_per_m <= h_px:
            return z
    return 1


def map_axes(ax, points, aspect, pad=0.15, min_span_m=4000, w_px=2000):
    """Basemap framing a list of (lat, lon) points inside axes of the given width/height aspect."""
    xy = [merc(*p) for p in points]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    w = max(max(xs) - min(xs), min_span_m) * (1 + 2 * pad)
    h = max(max(ys) - min(ys), min_span_m / aspect) * (1 + 2 * pad)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    zoom = _fit_zoom(w, h, w_px, w_px / aspect)
    draw_basemap(ax, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, zoom)
    return zoom


def _finish(fig, pdf):
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_bytes(build):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        build(pdf)
    return buf.getvalue()


# ---------------------------------------------------------------- main page
def wind_pdf(point, zoom, wind, tides=None, height_unit="m"):
    """One A4 page: map with the selected point, the wind charts and (if given) the tide plots."""
    from meteo import wind_mpl_figure
    from tides import tide_figures

    def build(pdf):
        fig = Figure(figsize=A4)
        fig.text(0.06, 0.975, "Typical winds", fontsize=15, fontweight="bold", va="top")
        fig.text(0.06, 0.953, f"Selected point: {point[0]:.4f}, {point[1]:.4f}  —  {wind['title']}",
                 fontsize=8, va="top")
        rect = (0.06, 0.665, 0.88, 0.265)  # map; the rows below are placed with a full-page grid
        ax = fig.add_axes(rect)
        cx, cy = merc(*point)
        aspect = rect[2] * A4[0] / (rect[3] * A4[1])
        z = int(min(max(zoom, 3), 13))
        half_w = 1.5 * 2 * math.pi * R_EARTH / 2 ** z  # the view the web map shows...
        half_h = half_w / aspect
        draw_basemap(ax, cx - half_w, cy - half_h, cx + half_w, cy + half_h, z + 1)  # ...from sharper tiles
        ax.plot(cx, cy, marker="o", ms=11, mfc="red", mec="white", mew=2, zorder=10)
        # (a GridSpec's margins are ignored by subfigures, hence full-page row ratios.)
        # rows from the top: header + map | wind | tides | footer
        gs = fig.add_gridspec(4, 2, height_ratios=[0.35, 0.24, 0.32, 0.09], width_ratios=[1, 4.5], hspace=0, wspace=0)
        wind_mpl_figure(wind, fig=fig.add_subfigure(gs[1, :]))
        if tides is not None:
            tide_figures(tides, height_unit, figs=(fig.add_subfigure(gs[2, 0]), fig.add_subfigure(gs[2, 1])))
        fig.text(0.06, 0.05, "Wind speeds in kn; wedge width is proportional to how common the wind is."
                 + (f" Tides: {tides['source']}; red = flood, blue = ebb; times in {tides['tzlabel']}."
                    if tides is not None else ""), fontsize=7, color="dimgray", va="top")
        _finish(fig, pdf)
    return _pdf_bytes(build)


# ---------------------------------------------------------------- route page
def _arrow_mpl(ax, lat, lon, e, n, scale, color, label="", z=5):
    """Arrow for a current/velocity (e, n) in kn; scale = degrees of latitude per knot (as in the web map)."""
    spd = math.hypot(e, n)
    x, y = merc(lat, lon)
    ax.plot(x, y, "o", ms=2.5, color="black", zorder=z)
    if label:
        ax.annotate(label.replace("<br>", "\n"), (x, y), xytext=(4, -4), textcoords="offset points",
                    fontsize=6.5, fontweight="bold", va="top", path_effects=HALO, zorder=z + 5)
    if spd < 0.02:
        ax.plot(x, y, "o", ms=9, mfc="none", mec=color, zorder=z)
        return
    c = math.cos(math.radians(lat)) or 1e-6
    ux, uy = e / spd, n / spd
    tip = (lat + uy * spd * scale, lon + ux * spd * scale / c)
    tx, ty = merc(*tip)
    ax.annotate("", xy=(tx, ty), xytext=(x, y), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=11, shrinkA=0, shrinkB=0))


def departure_mpl(fig, ok, top, use_local):
    ax = fig.add_subplot()
    ax.plot([t["departure"] for t in ok], [t["duration_h"] for t in ok], "-o", color="gray", mfc="red", mec="red",
            lw=1.2, ms=3.5, label="Duration")
    ax.plot([top["departure"]], [top["duration_h"]], "*", color="green", ms=14, label="Best")
    ax.annotate(f"{top['departure']:%H:%M}", (top["departure"], top["duration_h"]), xytext=(0, 8),
                textcoords="offset points", ha="center", color="green", fontsize=9)
    ax.set_ylabel("Duration (h)")
    ax.set_title(f"Trip duration vs departure time — {'local time' if use_local else 'UTC'}", loc="left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%a %d"))
    fig.subplots_adjust(left=0.1, right=0.94, bottom=0.2, top=0.88)


def route_pdf(res, dep_text, speed_text, mode_text, best=None):
    """PDF with the route map (current arrows), departure/arrival/speed and the best-departure plot.

    res: result dict of the route page; best: (ok_trials, top_trial, use_local, caption) or None.
    """
    def build(pdf):
        fig = Figure(figsize=A4)
        fig.text(0.06, 0.965, "Tidal currents along a route", fontsize=16, fontweight="bold", va="top")
        arrival = res["arrival"][2].replace("Arrival ", "") if res.get("arrival") else "n/a"
        lines = [f"Departure: {dep_text}", f"Arrival: {arrival}", f"{mode_text}: {speed_text}", res["summary"]]
        for i, t in enumerate(lines):
            fig.text(0.06, 0.935 - i * 0.017, t, fontsize=9, va="top")
        has_best = best is not None
        rect = (0.06, 0.40, 0.88, 0.44) if has_best else (0.06, 0.10, 0.88, 0.74)
        ax = fig.add_axes(rect)
        arrows = res["arrows"] + res.get("props", [])
        pts = list(res["pts"]) + [(a[0], a[1]) for a in arrows]
        zoom = map_axes(ax, pts, rect[2] * A4[0] / (rect[3] * A4[1]), pad=0.12)
        ax.plot(*zip(*[merc(*p) for p in res["pts"]]), "--", color="blue", lw=1.2, zorder=3)
        for a in res["arrows"]:
            _arrow_mpl(ax, a[0], a[1], a[2], a[3], a[4], "red", a[6])
        for a in res.get("props", []):
            _arrow_mpl(ax, a[0], a[1], a[2], a[3], a[4], "limegreen", z=6)
        for i, p in enumerate(res["pts"]):
            last = i == len(res["pts"]) - 1
            x, y = merc(*p)
            ax.plot(x, y, marker="o", ms=10, mfc="green" if i == 0 else ("red" if last else "orange"),
                    mec="white", mew=1.5, zorder=12)
            ax.annotate("Start" if i == 0 else ("End" if last else f"WP{i}"), (x, y), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, fontweight="bold", path_effects=HALO, zorder=13)
        if res.get("arrival"):
            x, y = merc(res["arrival"][0], res["arrival"][1])
            ax.annotate(res["arrival"][2], (x, y), xytext=(8, 16), textcoords="offset points", fontsize=8,
                        fontweight="bold", color="#7a0000", path_effects=HALO, zorder=14)
        fig.text(0.06, rect[1] - 0.012, "Red arrow: direction the water flows towards (length proportional to speed)."
                 + (" Green arrow: propulsion velocity through the water." if res.get("props") else ""),
                 fontsize=7, color="dimgray", va="top")
        if has_best:
            ok, top, use_local, caption = best
            fig.text(0.06, 0.34, caption, fontsize=8, va="top")
            departure_mpl(fig.add_subfigure(fig.add_gridspec(3, 1, height_ratios=[0.68, 0.28, 0.04], hspace=0)[1]), ok, top, use_local)
        _finish(fig, pdf)
    return _pdf_bytes(build)
