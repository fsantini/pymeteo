import streamlit as st
import folium
from matplotlib import pyplot as plt
from streamlit_folium import st_folium
import datetime
from datetime import date
from meteo import plot_winds
from tides import tide_figures, get_tides, available_sources, NoDataError

st.title("Click on the map to get the typical winds")

st.write("This app displays the typical winds for a given location and date. Click on the map to select a location, then choose a date from the calendar. The app will display the typical wind speeds and directions for that location and date.")
st.write("Data from meteostat. Map from Folium (OpenStreetMap).")

explanation_placeholder = st.empty()

def get_pos(lat,lng):
    return lat,lng

placeholder = st.empty()

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
    tcol1, tcol2 = st.columns(2)
    tide_source = tcol1.selectbox("Tide data source", available_sources())
    tide_tz = tcol2.radio("Times", ["Local time", "UTC"], horizontal=True)

explanation_placeholder.write(f"Speeds are in kn. The width of the wedges is proportional to how common the wind is. Winds below {cutoff_speed} knots are not displayed and the percentage of the time when the wind is below {cutoff_speed} knots is displayed in the circle in the middle of the plot.")

# Create a Folium map centered on a default location
map_center = [0, 0]
zoom_level = 2
world_map = folium.Map(location=map_center, zoom_start=zoom_level)

world_map.add_child(folium.LatLngPopup())
# Display the map using Streamlit's folium_static function
map = st_folium(world_map, height=300, width=600)

if map['last_clicked']:
    data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])

    with placeholder, st.spinner('Loading data...'):
        success = False
        try:
            plot_winds(data, selected_date, cutoff_speed)
            success = True
        except:
            pass

    if success:
        placeholder.pyplot(plt.gcf())
    else:
        placeholder.error("An error occurred. Please try a different coordinate.")

    if show_tides:
        st.subheader("Tidal currents (selected day ± 3 days)")
        try:
            with st.spinner('Computing tides...'):
                tides = get_tides(tide_source, data[0], data[1], selected_date, tide_tz == "Local time")
                compass_fig, tide_fig = tide_figures(tides)
            c1, c2 = st.columns([1, 3])
            c1.pyplot(compass_fig)
            c2.pyplot(tide_fig)
            st.caption(f"Source: {tides['source']}. Red arrow: flood current direction. "
                       f"Blue arrow: ebb current direction. Times in {tides['tzlabel']}.")
        except NoDataError as e:
            st.info(f"{tide_source} does not provide tidal current data at or near this location. {e}")
        except Exception as e:
            st.error(f"Could not compute tidal currents: {e}")

st.write("App by [Francesco Santini](https://francescosantini.com/). Code on [Github](https://github.com/fsantini/pymeteo).")