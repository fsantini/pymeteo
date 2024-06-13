import streamlit as st
import folium
from matplotlib import pyplot as plt
from streamlit_folium import st_folium
import datetime
from datetime import date
from meteo import plot_winds

st.title("Click on the map to get the typical winds")

st.write("This app displays the typical winds for a given location and date. Click on the map to select a location, then choose a date from the calendar. The app will display the typical wind speeds and directions for that location and date.")
st.write("Data from meteostat. Map from Folium (OpenStreetMap).")
st.write("Speeds are in kn. The width of the wedges is proportional to how common the wind is. Winds below 5 knots are not displayed and the percentage of the time when the wind is below 5 knots is displayed in the circle in the middle of the plot.")
def get_pos(lat,lng):
    return lat,lng

placeholder = st.empty()

# Create a Folium map centered on a default location
map_center = [0, 0]
zoom_level = 2
world_map = folium.Map(location=map_center, zoom_start=zoom_level)

world_map.add_child(folium.LatLngPopup())
# Display the map using Streamlit's folium_static function
map = st_folium(world_map, height=300, width=600)

# Get today's date
today = date.today()

# Display the date picker, default to today's date
selected_date = st.date_input("Select a date for the prediction", value=today)

if map['last_clicked']:
    data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])

    with placeholder, st.spinner('Loading data...'):
        success = False
        try:
            plot_winds(data, selected_date)
            success = True
        except:
            pass

    if success:
        placeholder.pyplot(plt.gcf())
    else:
        placeholder.error("An error occurred. Please try a different coordinate.")

