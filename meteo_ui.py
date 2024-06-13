import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QCalendarWidget
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from meteo import plot_winds

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("World Map and Calendar")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a vertical layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create a figure and add a subplot
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Draw the world map
        self.ax = self.figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        self.ax.add_feature(cfeature.LAND, facecolor='#C2B280')
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.BORDERS)

        self.location_marker = self.ax.plot([], [], marker='o', color='red', transform=ccrs.PlateCarree())[0]

        # Add the canvas to the layout
        layout.addWidget(self.canvas)

        # Create a calendar widget
        self.calendar = QCalendarWidget(self)
        layout.addWidget(self.calendar)

        # Connect the event
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_drag_end)

        # Initialize dragging variables
        self.dragging = False
        self.last_x, self.last_y = None, None

    def on_drag_end(self, event):
        self.dragging = False

    def on_drag(self, event):
        if self.dragging and event.inaxes:
            dx = event.xdata - self.last_x
            dy = event.ydata - self.last_y
            self.last_x, self.last_y = event.xdata, event.ydata
            self.ax.set_xlim(self.ax.get_xlim() - dx)
            self.ax.set_ylim(self.ax.get_ylim() - dy)
            self.canvas.draw()

    def on_scroll(self, event):
        print(event.button)
        if event.inaxes:
            ax = event.inaxes
            xdata, ydata = event.xdata, event.ydata
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            zoom_factor = 1.5 if event.button == 'up' else 1 / 1.5

            new_width = (x_max - x_min) * 1 / zoom_factor
            new_height = (y_max - y_min) * 1 / zoom_factor

            relx = (xdata - x_min) / (x_max - x_min)
            rely = (ydata - y_min) / (y_max - y_min)

            x_min = xdata - new_width * relx
            x_max = xdata + new_width * (1 - relx)
            y_min = ydata - new_height * rely
            y_max = ydata + new_height * (1 - rely)

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            self.canvas.draw()

    def on_click(self, event):
        print(event.button)
        if event.inaxes:
            if event.button == 1:
                x, y = event.xdata, event.ydata
                self.location_marker.set_data([x], [y])
                self.canvas.draw()

                # Trigger event here, like printing coordinates or selected date
                print(f"Coordinates: {x}, {y}")
                print(f"Selected Date: {self.calendar.selectedDate().toString()}")
                plot_winds((y,x), self.calendar.selectedDate().toPyDate())
                plt.gcf().show()
            elif event.button == 2:
                print('dragging')
                self.dragging = True
                self.last_x, self.last_y = event.xdata, event.ydata


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())