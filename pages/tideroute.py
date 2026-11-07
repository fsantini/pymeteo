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
from tides import available_sources, current_vectors, local_timezone, NoDataError  # noqa: E402

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


def distance_nm(p1, p2):
    la1, lo1, la2, lo2 = map(math.radians, (*p1, *p2))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_NM * math.asin(math.sqrt(h))


def arrow(m, lat, lon, e, n, scale, tooltip, label=""):
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
        folium.CircleMarker((lat, lon), radius=5, color="red", fill=False, tooltip=tooltip).add_to(m)
        return
    ux, uy = e / spd, n / spd  # unit vector (east, north)
    length = spd * scale
    tip = (lat + uy * length, lon + ux * length / c)
    head = length * 0.3
    pts = []
    for ang in (150, -150):  # arrowhead barbs
        a = math.radians(ang)
        bx, by = ux * math.cos(a) - uy * math.sin(a), ux * math.sin(a) + uy * math.cos(a)
        pts.append((tip[0] + by * head, tip[1] + bx * head / c))
    folium.PolyLine([(lat, lon), tip], color="red", weight=3, tooltip=tooltip).add_to(m)
    folium.PolyLine([pts[0], tip, pts[1]], color="red", weight=3, tooltip=tooltip).add_to(m)


# ---- inputs ----
if "route_pts" not in st.session_state:
    st.session_state.route_pts = []
    st.session_state.last_click = None
pts = st.session_state.route_pts

c1, c2, c3 = st.columns(3)
dep_date = c1.date_input("Departure date", value=datetime.now().date())
dep_time = c2.time_input("Departure time", value=time(9, 0), step=900)
speed_kn = c3.number_input("Boat speed (kn)", min_value=0.5, max_value=40.0, value=5.0, step=0.5)
c4, c5, c6 = st.columns(3)
n_points = c4.number_input("Route points (start, waypoints, end)", min_value=2, max_value=12, value=2, step=1)
n_steps = c5.slider("Number of steps", 2, 60, 10)
source = c6.selectbox("Tide data source", available_sources())
del pts[n_points:]  # fewer points requested than already placed

def clear_route():
    """Forget the points, the computed route/arrows and the map's frontend state (popups etc.)."""
    st.session_state.route_pts = []
    st.session_state.last_click = None
    st.session_state.pop("applied", None)
    st.session_state.pop("result", None)
    st.session_state.map_gen = st.session_state.get("map_gen", 0) + 1  # new map key = fresh map


if st.button("Reset points"):
    clear_route()
    st.rerun()

def compute(pts, dep_date, dep_time, speed_kn, n_steps, source):
    """Tidal currents along the route; returns a dict with arrows, table rows and a message."""
    legs = [distance_nm(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(legs)
    tz = local_timezone(*pts[0])
    dep = datetime.combine(dep_date, dep_time, tzinfo=tz)
    fr = [i / n_steps for i in range(n_steps + 1)]
    dists = [total * f for f in fr]
    places = [point_at(pts, legs, d) for d in dists]
    when = [dep + timedelta(hours=d / speed_kn) for d in dists]
    utc = np.array([np.datetime64(w.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)) for w in when])
    res = dict(pts=list(pts), arrows=[], rows=[], error=None, info=None,
               summary=f"Route length: {total:.1f} nm — duration {total / speed_kn:.1f} h — "
                       f"times in local time ({tz.key}).")
    try:
        vec = current_vectors(source, [p[0] for p in places], [p[1] for p in places], utc)
        valid = [v for v in vec if v]
        if not valid:
            raise NoDataError("No tidal current data along this route.")
        # arrow length: strongest current spans ~ one step
        step_deg = (total / n_steps) / 60.0
        scale = 0.9 * step_deg / max(max(math.hypot(*v) for v in valid), 0.5)
        for (lat, lon), d, w, v in zip(places, dists, when, vec):
            if v is None:
                continue
            spd, brg = math.hypot(*v), math.degrees(math.atan2(v[0], v[1])) % 360
            res["arrows"].append((lat, lon, v[0], v[1], scale,
                                  f"{w:%d %b %H:%M} — {spd:.2f} kn towards {brg:.0f}°",
                                  f"{w:%H:%M}<br>{spd:.1f} kn"))
            res["rows"].append({"Distance (nm)": round(d, 1), "Time (local)": f"{w:%d %b %H:%M}",
                                "Current (kn)": round(spd, 2), "Towards (°)": round(brg)})
    except NoDataError as e:
        res["info"] = f"{source} does not provide tidal current data for this route. {e}"
    except Exception as e:
        res["error"] = f"Could not compute tidal currents: {e}"
    return res


# ---- recalculation ----
params = (tuple(pts), dep_date, dep_time, speed_kn, n_steps, source)
ready = len(pts) == n_points
applied = st.session_state.get("applied")
# fixed slots keep the map at the same position in the page, so it is never remounted
slot_top = st.container()
slot_status = st.container()
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
    st.caption("Arrow: direction the water flows towards; length proportional to current speed.")
