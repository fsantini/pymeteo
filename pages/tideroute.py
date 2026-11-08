import math
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdf_export import route_pdf  # noqa: E402
from tides import PLOT_CONFIG, available_sources, current_vectors, local_timezone, NoDataError  # noqa: E402

R_NM = 3440.065  # earth radius in nautical miles

st.set_page_config(page_title="Tide route")
st.title("Tidal currents along a route")
st.write("Click twice on the map to choose the start and end points of a straight route. "
         "Choose departure time and boat speed: the tidal current is computed at the time the boat "
         "is expected to pass each step and shown as an arrow (length proportional to speed).")
st.page_link("meteo_streamlit.py", label="← Back to the wind app")


def interpolate(p1, p2, f):
    """Point at fraction f along the great circle p1 -> p2 (lat, lon degrees)."""
    la1, lo1, la2, lo2 = map(math.radians, (*p1, *p2))
    d = distance_nm(p1, p2) / R_NM
    if d == 0:
        return p1
    a, b = math.sin((1 - f) * d) / math.sin(d), math.sin(f * d) / math.sin(d)
    x = a * math.cos(la1) * math.cos(lo1) + b * math.cos(la2) * math.cos(lo2)
    y = a * math.cos(la1) * math.sin(lo1) + b * math.cos(la2) * math.sin(lo2)
    z = a * math.sin(la1) + b * math.sin(la2)
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


def point_at(pts, legs, d):
    """Position after d nautical miles along the multi-leg route."""
    for a, b, leg in zip(pts, pts[1:], legs):
        if d <= leg or b is pts[-1]:
            return interpolate(a, b, min(d / leg, 1.0) if leg else 0.0)
        d -= leg


def bearing_deg(p1, p2):
    """Initial great-circle bearing from p1 to p2, degrees clockwise from north."""
    la1, lo1, la2, lo2 = map(math.radians, (*p1, *p2))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return math.degrees(math.atan2(y, x)) % 360


def distance_nm(p1, p2):
    la1, lo1, la2, lo2 = map(math.radians, (*p1, *p2))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_NM * math.asin(math.sqrt(h))


def arrival_label(m, lat, lon, text):
    """Bold label above the final point (the step label sits below-right of it)."""
    html = ('<div style="font-size:12px;font-weight:700;white-space:nowrap;color:#7a0000;'
            'text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;">'
            f'{text}</div>')
    folium.Marker((lat, lon), icon=folium.DivIcon(html=html, icon_size=(120, 16),
                                                  icon_anchor=(-10, 30))).add_to(m)


def arrow(m, lat, lon, e, n, scale, tooltip, label="", color="red"):
    """Draw an arrow at (lat, lon) for a current (e, n) in kn; scale = degrees of latitude per knot."""
    spd = math.hypot(e, n)
    c = math.cos(math.radians(lat)) or 1e-6
    folium.CircleMarker((lat, lon), radius=2, color="black", fill=True, weight=1).add_to(m)
    if label:
        html = ('<div style="font-size:11px;font-weight:600;line-height:1.1;white-space:nowrap;'
                'color:#111;text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;">'
                f'{label}</div>')
        folium.Marker((lat, lon), icon=folium.DivIcon(html=html, icon_size=(60, 26),
                                                      icon_anchor=(-5, -4))).add_to(m)
    if spd < 0.02:
        folium.CircleMarker((lat, lon), radius=5, color=color, fill=False, tooltip=tooltip).add_to(m)
        return
    ux, uy = e / spd, n / spd  # unit vector (east, north)
    length = spd * scale
    tip = (lat + uy * length, lon + ux * length / c)
    head = length * 0.2
    pts = []
    for ang in (170, -170):  # arrowhead barbs
        a = math.radians(ang)
        bx, by = ux * math.cos(a) - uy * math.sin(a), ux * math.sin(a) + uy * math.cos(a)
        pts.append((tip[0] + by * head, tip[1] + bx * head / c))
    folium.PolyLine([(lat, lon), tip], color=color, weight=3, tooltip=tooltip).add_to(m)
    folium.PolyLine([pts[0], tip, pts[1]], color=color, weight=3, tooltip=tooltip).add_to(m)


# ---- inputs ----
if "route_pts" not in st.session_state:
    st.session_state.route_pts = []
    st.session_state.last_click = None
pts = st.session_state.route_pts

MODES = ["Constant VMG (speed over ground)", "Constant propulsion (speed through water)"]
mode = st.radio("Mode", MODES, index=1, horizontal=True)
propulsion = mode == MODES[1]
use_local = st.radio("Times (departure input and results)", ["Local time (at the start point)", "UTC"],
                     horizontal=True).startswith("Local")
c1, c2, c3 = st.columns(3)
dep_date = c1.date_input("Departure date", value=datetime.now().date(), key="dep_date")
dep_time = c2.time_input("Departure time", value=time(9, 0), step=900, key="dep_time")
speed_kn = c3.number_input("Propulsion speed through water (kn)" if propulsion else "Boat speed over ground (kn)", min_value=0.5, max_value=40.0, value=5.0, step=0.5)
c4, c5, c6 = st.columns(3)
n_points = c4.number_input("Route points (start, waypoints, end)", min_value=2, max_value=12, value=2, step=1)
n_steps = c5.slider("Number of steps", 2, 60, 10)
source = c6.selectbox("Tide data source", available_sources())
del pts[n_points:]  # fewer points requested than already placed

def blend(v1, v2, f):
    return (v1[0] * (1 - f) + v2[0] * f, v1[1] * (1 - f) + v2[1] * f)


def fill_gaps(vec, dists):
    """Fill points without data by linear interpolation (by distance) between the previous and next
    point that has data. Gaps at either end of the route stay empty. Returns (vec, set of filled indices)."""
    vec, filled = list(vec), set()
    known = [i for i, v in enumerate(vec) if v is not None]
    for i, v in enumerate(vec):
        if v is not None:
            continue
        before = [k for k in known if k < i]
        after = [k for k in known if k > i]
        if before and after:
            p, q = before[-1], after[0]
            vec[i] = blend(vec[p], vec[q], (dists[i] - dists[p]) / (dists[q] - dists[p]))
            filled.add(i)
    return vec, filled


def clear_route():
    """Forget the points, the computed route/arrows and the map's frontend state (popups etc.)."""
    st.session_state.route_pts = []
    st.session_state.last_click = None
    st.session_state.pop("applied", None)
    st.session_state.pop("result", None)
    st.session_state.map_gen = st.session_state.get("map_gen", 0) + 1  # new map key = fresh map


rb1, rb2, _ = st.columns([1, 1, 3])
if rb1.button("Reset points"):
    clear_route()
    st.rerun()
if rb2.button("Reverse route", disabled=len(pts) < 2):
    st.session_state.route_pts = pts[::-1]  # start <-> end, waypoints in reverse order
    st.rerun()

def compute_propulsion(pts, dep_date, dep_time, speed_kn, n_steps, source, use_local=True):
    """Constant propulsion speed through the water; heading corrected for the current.

    Walks along the route step by step: at each point the current at the time of passage gives the
    cross-track component to cancel (course correction) and the along-track component that adds to
    the along-track part of the propulsion, i.e. the VMG used to reach the next point.
    """
    legs = [distance_nm(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(legs)
    tz = local_timezone(*pts[0]) if use_local else ZoneInfo("UTC")
    tcol = "Time (local)" if use_local else "Time (UTC)"
    tlabel = f"local time, {tz.key}" if use_local else "UTC"
    dep = datetime.combine(dep_date, dep_time, tzinfo=tz)
    # step points plus the intermediate waypoints, so every segment lies on a single leg
    cum = np.cumsum(legs)
    marks = {round(total * i / n_steps, 6): None for i in range(n_steps + 1)}
    for k in range(1, len(pts) - 1):
        marks[round(float(cum[k - 1]), 6)] = pts[k]
    dists = sorted(marks)
    places = [marks[d] or point_at(pts, legs, d) for d in dists]
    res = dict(pts=list(pts), arrows=[], props=[], rows=[], error=None, info=None, summary="", duration_h=None, arrival=None)
    t, missing, filled, last_valid = dep, 0, 0, None
    # one scale for both arrows; the propulsion arrow (largest) spans about one step
    scale = 0.7 * (total / n_steps / 60.0) / max(speed_kn, 0.5)
    try:
        for i, ((lat, lon), d) in enumerate(zip(places, dists)):
            utc = np.datetime64(t.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))
            v = current_vectors(source, [lat], [lon], [utc])[0]
            mark = ""
            if v is not None:
                last_valid = (d, v)
            else:
                # no data here: interpolate between the previous point with data and the next one,
                # whose passage time is estimated assuming the propulsion speed over the ground
                nxt = None
                if last_valid:
                    for j in range(i + 1, len(places)):
                        tj = t + timedelta(hours=(dists[j] - d) / speed_kn)
                        vj = current_vectors(source, [places[j][0]], [places[j][1]],
                                             [np.datetime64(tj.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))])[0]
                        if vj is not None:
                            nxt = (dists[j], vj)
                            break
                if last_valid and nxt:
                    v = blend(last_valid[1], nxt[1], (d - last_valid[0]) / (nxt[0] - last_valid[0]))
                    mark, filled = "~", filled + 1
                else:
                    missing += 1
            ce, cn = v or (0.0, 0.0)
            cspd, cbrg = math.hypot(ce, cn), math.degrees(math.atan2(ce, cn)) % 360
            row = {"Distance (nm)": round(d, 1), tcol: f"{t:%d %b %H:%M}",
                   "Current (kn)": round(cspd, 2), "Towards (°)": round(cbrg)}
            if v is not None:
                res["arrows"].append((lat, lon, ce, cn, scale,
                                      f"{t:%d %b %H:%M} — current {mark}{cspd:.2f} kn towards {cbrg:.0f}°"
                                      + (" (interpolated)" if mark else ""),
                                      f"{t:%H:%M}<br>{mark}{cspd:.1f} kn"))
            if i == len(places) - 1:
                res["rows"].append(row)
                break
            trk = bearing_deg((lat, lon), places[i + 1])
            tx, ty = math.sin(math.radians(trk)), math.cos(math.radians(trk))  # track unit (E, N)
            along = ce * tx + cn * ty
            cross = ce * ty - cn * tx  # component to the right of the track
            where = f"at {lat:.3f}, {lon:.3f} ({t:%d %b %H:%M}, current {cspd:.1f} kn towards {cbrg:.0f}°)"
            if abs(cross) >= speed_kn:
                raise ValueError(f"Propulsion speed {speed_kn:g} kn is too low to hold the course {where}: "
                                 f"the cross-current is {abs(cross):.1f} kn.")
            w_along, w_cross = math.sqrt(speed_kn ** 2 - cross ** 2), -cross
            vmg = w_along + along
            if vmg <= 0:
                raise ValueError(f"Propulsion speed {speed_kn:g} kn is too low {where}: "
                                 f"VMG is {vmg:.2f} kn (<= 0), the boat cannot make progress.")
            # propulsion velocity relative to the water (E, N) and heading
            pe, pn = w_along * tx + w_cross * ty, w_along * ty - w_cross * tx
            head = math.degrees(math.atan2(pe, pn)) % 360
            corr = (head - trk + 180) % 360 - 180
            res["props"].append((lat, lon, pe, pn, scale,
                                 f"Propulsion {speed_kn:g} kn, heading {head:.0f}° "
                                 f"(correction {corr:+.0f}° from track {trk:.0f}°), VMG {vmg:.2f} kn"))
            row.update({"Heading (°)": round(head), "Correction (°)": round(corr), "VMG (kn)": round(vmg, 2)})
            res["rows"].append(row)
            t += timedelta(hours=(dists[i + 1] - d) / vmg)
        res["summary"] = (f"Route length: {total:.1f} nm — duration {(t - dep).total_seconds() / 3600:.1f} h — "
                          f"times in {tlabel}.")
        res["duration_h"] = (t - dep).total_seconds() / 3600
        res["arrival"] = (*places[-1], f"Arrival {t:%d %b %H:%M}")
        notes = []
        if filled:
            notes.append(f"{filled} point(s) without {source} data use a current interpolated between the "
                         "previous and next point (marked ~)")
        if missing:
            notes.append(f"{missing} point(s) have no data to interpolate from: zero current assumed there")
        if notes:
            res["info"] = "; ".join(notes) + "."
    except NoDataError as e:
        res["info"] = f"{source} does not provide tidal current data for this route. {e}"
    except ValueError as e:
        res["error"] = str(e)
    except Exception as e:
        res["error"] = f"Could not compute tidal currents: {e}"
    return res


def compute(pts, dep_date, dep_time, speed_kn, n_steps, source, propulsion=False, use_local=True):
    if propulsion:
        return compute_propulsion(pts, dep_date, dep_time, speed_kn, n_steps, source, use_local)
    return compute_vmg(pts, dep_date, dep_time, speed_kn, n_steps, source, use_local)


def compute_vmg(pts, dep_date, dep_time, speed_kn, n_steps, source, use_local=True):
    """Tidal currents along the route; returns a dict with arrows, table rows and a message."""
    legs = [distance_nm(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(legs)
    tz = local_timezone(*pts[0]) if use_local else ZoneInfo("UTC")
    tcol = "Time (local)" if use_local else "Time (UTC)"
    tlabel = f"local time, {tz.key}" if use_local else "UTC"
    dep = datetime.combine(dep_date, dep_time, tzinfo=tz)
    fr = [i / n_steps for i in range(n_steps + 1)]
    dists = [total * f for f in fr]
    places = [point_at(pts, legs, d) for d in dists]
    when = [dep + timedelta(hours=d / speed_kn) for d in dists]
    utc = np.array([np.datetime64(w.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)) for w in when])
    res = dict(pts=list(pts), arrows=[], rows=[], error=None, info=None,
               arrival=(*places[-1], f"Arrival {when[-1]:%d %b %H:%M}"),
               summary=f"Route length: {total:.1f} nm — duration {total / speed_kn:.1f} h — "
                       f"times in {tlabel}.")
    try:
        vec = current_vectors(source, [p[0] for p in places], [p[1] for p in places], utc)
        valid = [v for v in vec if v]
        if not valid:
            raise NoDataError("No tidal current data along this route.")
        vec, filled = fill_gaps(vec, dists)
        valid = [v for v in vec if v]
        if filled:
            res["info"] = (f"{source} has no current data at {len(filled)} point(s) on the route: "
                           "the current there is interpolated between the previous and next point (marked ~).")
        # arrow length: strongest current spans ~ one step
        step_deg = (total / n_steps) / 60.0
        scale = 0.9 * step_deg / max(max(math.hypot(*v) for v in valid), 0.5)
        for i, ((lat, lon), d, w, v) in enumerate(zip(places, dists, when, vec)):
            if v is None:
                continue
            spd, brg = math.hypot(*v), math.degrees(math.atan2(v[0], v[1])) % 360
            mark = "~" if i in filled else ""
            res["arrows"].append((lat, lon, v[0], v[1], scale,
                                  f"{w:%d %b %H:%M} — {mark}{spd:.2f} kn towards {brg:.0f}°"
                                  + (" (interpolated)" if mark else ""),
                                  f"{w:%H:%M}<br>{mark}{spd:.1f} kn"))
            res["rows"].append({"Distance (nm)": round(d, 1), tcol: f"{w:%d %b %H:%M}",
                                "Current (kn)": round(spd, 2), "Towards (°)": round(brg)})
    except NoDataError as e:
        res["info"] = f"{source} does not provide tidal current data for this route. {e}"
    except Exception as e:
        res["error"] = f"Could not compute tidal currents: {e}"
    return res


# ---- recalculation ----
params = (tuple(pts), dep_date, dep_time, speed_kn, n_steps, source, propulsion, use_local)
ready = len(pts) == n_points
applied = st.session_state.get("applied")
# fixed slots keep the map at the same position in the page, so it is never remounted
slot_top = st.container()
slot_status = st.container()
slot_pdf = st.container()
if ready and applied is None:  # first route: calculate automatically
    run = True
elif ready and applied != params:
    with slot_top:
        st.warning("Parameters changed: the arrows on the map are out of date.")
        # shade the stale map instead of hiding it
        st.markdown("<style>iframe[title*='st_folium']{opacity:.4;transition:opacity .2s}</style>",
                    unsafe_allow_html=True)
        run = st.button("Recalculate", type="primary")
    if run:
        with st.spinner("Computing tidal currents..."):
            st.session_state.result = compute(*params)
        st.session_state.applied = params
        st.rerun()  # redraw without the (now unneeded) button
else:
    run = False
if run and applied is None:
    with st.spinner("Computing tidal currents..."):
        st.session_state.result = compute(*params)
    st.session_state.applied = applied = params
res = st.session_state.get("result")

# ---- map ----
m = folium.Map(location=pts[0] if pts else [0, 0], zoom_start=8 if pts else 2)
for i, p in enumerate(pts):
    last = i == n_points - 1
    folium.Marker(p, tooltip="Start" if i == 0 else ("End" if last else f"Waypoint {i}"),
                  icon=folium.Icon(color="green" if i == 0 else ("red" if last else "orange"))).add_to(m)
if res and not pts:
    res = None
if res:
    folium.PolyLine(res["pts"], color="blue", weight=2, dash_array="6").add_to(m)
    for a in res["arrows"]:
        arrow(m, *a[:5], tooltip=a[5], label=a[6])
    if res.get("arrival"):
        arrival_label(m, *res["arrival"])
    for a in res.get("props", []):
        arrow(m, *a[:5], tooltip=a[5], color="green")
if ready:
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(40, 40))
    if res:
        with slot_status:
            if res["info"]:
                st.info(res["info"])
            if res["error"]:
                st.error(res["error"])
            st.write(res["summary"])
else:
    m.add_child(folium.LatLngPopup())
    slot_status.write(f"Click on the map to select the {'start' if not pts else ('end' if len(pts) == n_points - 1 else f'waypoint {len(pts)}')} point "
                       f"({len(pts) + 1} of {n_points}).")

out = st_folium(m, height=450, width=700, returned_objects=["last_clicked"], key=f"routemap{st.session_state.get('map_gen', 0)}")
click = out.get("last_clicked") if out else None
if click:
    key = (click["lat"], click["lng"])
    if key != st.session_state.last_click:
        st.session_state.last_click = key
        if len(pts) >= n_points:  # starting a new route: drop the old one completely
            clear_route()
            st.session_state.last_click = key
            st.session_state.route_pts.append(key)
            st.rerun()
        pts.append(key)
        st.rerun()

if res and res["rows"]:
    st.dataframe(res["rows"], hide_index=True)
    st.caption("Red arrow: direction the water flows towards. "
               + ("Green arrow: propulsion velocity through the water (heading). Both to the same scale." if res.get("props") else "Length proportional to current speed."))


# ---- best departure time (constant propulsion only) ----
def best_departure(pts, first_dep, window_h, step_min, speed_kn, n_steps, source, use_local=True, progress=None):
    """Try departures every step_min minutes from first_dep (naive datetime in the chosen time zone) for window_h hours.

    Returns a list of dicts (departure, duration_h or None, error). Each trial is a full
    constant-propulsion simulation, so the current is always taken at the actual passage times.
    """
    n = int(window_h * 60 // step_min) + 1
    out = []
    for k in range(n):
        dt = first_dep + timedelta(minutes=k * step_min)
        r = compute_propulsion(pts, dt.date(), dt.time(), speed_kn, n_steps, source, use_local)
        out.append(dict(departure=dt, duration_h=r["duration_h"], error=r["error"]))
        if progress:
            progress((k + 1) / n)
    return out


def departure_figure(ok, top, use_local):
    """Interactive plot of the trip duration against the departure time (best one highlighted)."""
    import plotly.graph_objects as go
    hm = lambda h: f"{int(h)} h {round((h % 1) * 60):02d} min"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[t["departure"] for t in ok], y=[t["duration_h"] for t in ok], mode="lines+markers", name="Duration",
        line=dict(color="gray", width=1.5), marker=dict(size=6, color="red"),
        customdata=[hm(t["duration_h"]) for t in ok],
        hovertemplate="depart %{x|%H:%M}<br>%{y:.2f} h (%{customdata})<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[top["departure"]], y=[top["duration_h"]], mode="markers+text", name="Best",
        marker=dict(symbol="star", size=16, color="green"), text=[f"{top['departure']:%H:%M}"],
        textposition="top center", textfont=dict(size=12, color="green"),
        hovertemplate="BEST: depart %{x|%H:%M}<br>%{y:.2f} h<extra></extra>"))
    fig.update_layout(
        title=dict(text=f"Trip duration vs departure time — {'local time' if use_local else 'UTC'}",
                   x=0.01, xanchor="left", font=dict(size=13)),
        xaxis=dict(gridcolor="rgba(128,128,128,0.25)", tickformatstops=[
            dict(dtickrange=[None, 86400000], value="%H:%M<br>%a %d"),
            dict(dtickrange=[86400000, None], value="%a %d %b")]),
        yaxis=dict(title="Duration (h)", gridcolor="rgba(128,128,128,0.25)", fixedrange=True),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=60, b=10), height=380, dragmode="zoom", hovermode="closest")
    return fig


def use_departure(dt):
    st.session_state.dep_date, st.session_state.dep_time = dt.date(), dt.time()


if propulsion and ready:
    with st.expander("Find the best departure time"):
        b1, b2 = st.columns(2)
        window_h = b1.number_input("Window length from the departure time above (h)", 1.0, 72.0, 12.0, 1.0)
        step_min = b2.number_input("Try a departure every (min)", 5, 240, 30, 5)
        if st.button("Find best departure"):
            bar = st.progress(0.0)
            first = datetime.combine(dep_date, dep_time)
            st.session_state.best = dict(
                trials=best_departure(pts, first, window_h, step_min, speed_kn, n_steps, source, use_local, bar.progress),
                key=(tuple(pts), first, window_h, step_min, speed_kn, n_steps, source, use_local), use_local=use_local)
            bar.empty()
        best = st.session_state.get("best")
        if best:
            ok = [t for t in best["trials"] if t["duration_h"] is not None]
            if not ok:
                st.error("No departure in the window allows completing the route: "
                         + (best["trials"][0]["error"] or "no data."))
            else:
                top = min(ok, key=lambda t: t["duration_h"])
                worst = max(ok, key=lambda t: t["duration_h"])
                st.success(f"Best departure: {top['departure']:%a %d %b %H:%M} — duration "
                           f"{top['duration_h']:.2f} h (worst in window: {worst['duration_h']:.2f} h at "
                           f"{worst['departure']:%H:%M}, difference {(worst['duration_h'] - top['duration_h']) * 60:.0f} min).")
                st.button("Use this departure time", on_click=use_departure, args=(top["departure"],))
                if len(ok) < len(best["trials"]):
                    st.caption(f"{len(best['trials']) - len(ok)} departure(s) skipped: propulsion too low for the current.")
                st.plotly_chart(departure_figure(ok, top, best.get("use_local", True)), width="stretch",
                                config=PLOT_CONFIG)
            st.caption(f"Departure times are in {'local time at the start point' if best.get('use_local', True) else 'UTC'}. Results are not updated "
                       "automatically when parameters change: run the search again.")


# ---- PDF export (filled last, into its slot near the map, so it sees the departure search above) ----
def current_best():
    """(ok trials, best trial, use_local, caption) of the departure search, if it matches the current route."""
    b = st.session_state.get("best")
    if not (propulsion and ready and b):
        return None
    k = b["key"]  # (pts, first, window_h, step_min, speed_kn, n_steps, source, use_local)
    if (k[0], k[4], k[5], k[6]) != (tuple(pts), speed_kn, n_steps, source):
        return None
    ok = [t for t in b["trials"] if t["duration_h"] is not None]
    if not ok:
        return None
    top = min(ok, key=lambda t: t["duration_h"])
    ul = b.get("use_local", True)
    return ok, top, ul, (f"Best departure: {top['departure']:%a %d %b %H:%M} — duration {top['duration_h']:.2f} h "
                         f"({'local time at the start point' if ul else 'UTC'}).")


if res and res["rows"] and applied == params:  # the PDF shows what is on the map: not when it is stale
    # everything is evaluated now: the download callback runs later, outside the script run
    pdf_args = (res, f"{datetime.combine(dep_date, dep_time):%a %d %b %Y %H:%M} "
                     f"({'local time' if use_local else 'UTC'})",
                f"{speed_kn:g} kn", "Propulsion speed through water" if propulsion else "Boat speed over ground",
                current_best())
    slot_pdf.download_button("Export to PDF", lambda: route_pdf(*pdf_args), file_name="route.pdf",
                             mime="application/pdf", on_click="ignore", icon=":material/picture_as_pdf:")
