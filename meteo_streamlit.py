import streamlit as st
import folium
from matplotlib import pyplot as plt
from streamlit_folium import st_folium
import datetime
from datetime import date
from meteo import wind_data, wind_figures, WEATHER_SOURCES
from tides import PLOT_CONFIG, tide_figures_plotly, get_tides, available_sources, NoDataError

st.title("Click on the map to get the typical winds")

st.write("This app displays the typical winds for a given location and date. Click on the map to select a location, then choose a date from the calendar. The app will display the typical wind speeds and directions for that location and date.")
st.write("Weather data from Meteostat or Open-Meteo. Map from Folium (OpenStreetMap).")

st.page_link("pages/tideroute.py", label="Tide route planner: tidal currents along a route →")

explanation_placeholder = st.empty()

def get_pos(lat,lng):
    return lat,lng

# Get today's date
today = date.today()

# Create two columns
col1, col2 = st.columns(2)

# In the left column, display the date picker
with col1:
    selected_date = st.date_input("Select a date for the prediction", value=today)

# In the right column, display the slider for selecting the cutoff wind speed
with col2:
    cutoff_speed = st.slider("Select a cutoff wind speed", min_value=0, max_value=30, value=5, step=1)

show_tides = st.checkbox("Show tidal currents")
tide_source = None
if show_tides:
    tcol1, tcol2, tcol3 = st.columns(3)
    tide_source = tcol1.selectbox("Tide data source", available_sources())
    tide_tz = tcol2.radio("Times", ["Local time", "UTC"], horizontal=True)
    height_unit = tcol3.radio("Tide height", ["m", "ft"], horizontal=True)

weather_source = st.selectbox("Weather data source", WEATHER_SOURCES)

explanation_placeholder.write(f"Speeds are in kn. The width of the wedges is proportional to how common the wind is. Winds below {cutoff_speed} knots are not displayed and the percentage of the time when the wind is below {cutoff_speed} knots is displayed in the circle in the middle of the plot.")

# Create a Folium map centered on a default location
map_center = [0, 0]
zoom_level = 2
world_map = folium.Map(location=map_center, zoom_start=zoom_level)

world_map.add_child(folium.LatLngPopup())
# Display the map using Streamlit's folium_static function
map = st_folium(world_map, height=300, width=600)

# wind plot goes right below the map, above the tidal section
placeholder = st.empty()

if map['last_clicked']:
    data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])

    with placeholder.container():
        try:
            with st.spinner('Loading data...'):
                wind = wind_data(data, selected_date, cutoff_speed, weather_source)
                figs = wind_figures(wind)
        except Exception:
            st.error("An error occurred. Please try a different coordinate.")
        else:
            st.markdown(f"**{wind['title']}**")
            wc1, wc2 = st.columns(2)  # side by side on desktop, stacked on phones
            wc1.plotly_chart(figs[0], width="stretch", config=PLOT_CONFIG)
            wc2.plotly_chart(figs[1], width="stretch", config=PLOT_CONFIG)
            st.caption(wind['footer'])

    if show_tides:
        st.subheader("Tidal currents (selected day ± 3 days)")
        try:
            with st.spinner('Computing tides...'):
                tides = get_tides(tide_source, data[0], data[1], selected_date, tide_tz == "Local time")
                compass_fig, tide_fig = tide_figures_plotly(tides, height_unit)
            c1, c2 = st.columns([1, 3])
            c1.plotly_chart(compass_fig, width="stretch", config=PLOT_CONFIG)
            c2.plotly_chart(tide_fig, width="stretch", config=PLOT_CONFIG)
            st.caption(f"Source: {tides['source']}. Red arrow: flood current direction. "
                       f"Blue arrow: ebb current direction. Times in {tides['tzlabel']}.")
        except NoDataError as e:
            st.info(f"{tide_source} does not provide tidal current data at or near this location. {e}")
        except Exception as e:
            st.error(f"Could not compute tidal currents: {e}")

st.write("App by [Francesco Santini](https://francescosantini.com/). Code on [Github](https://github.com/fsantini/pymeteo).")