#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:51:29 2023

@author: francesco
"""

from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from meteostat import Point, Daily, Hourly
import math
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

def plot_winds(coords, desired_date, min_speed = 5):
    location = Point(coords[0], coords[1], 0)
    location.radius = 100000
    location.method = 'weighted'
    now = datetime.now()
    current_year = now.year

    print('Desired date', desired_date)

    if isinstance(desired_date, date):
        desired_date = datetime(desired_date.year, desired_date.month, desired_date.day)

    month = desired_date.month
    day = desired_date.day

    # if date is in the past, use current year for averaging
    if desired_date < now:
        starting_year = current_year
    else:
        starting_year = current_year - 1

    def get_data_for_year(year, month, day, point):
        midpoint = datetime(year, month, day)
        start = midpoint - timedelta(days=3)
        end = midpoint + timedelta(days=3)
        data = Hourly(point, start, end)
        data = data.fetch()
        return data

    wind_speeds = []
    wind_dirs = []
    temps = []
    precip = []

    for year in range(starting_year-average_years+1, starting_year+1):
        d = get_data_for_year(year, month, day, location)
        wind_speeds.extend(d['wspd'])
        wind_dirs.extend(d['wdir'])
        temps.extend(d['temp'].dropna())
        precip.extend(d['prcp'].dropna())


    # sanitize wind
    for i in range(len(wind_speeds)-1,-1,-1):
        if math.isnan(wind_speeds[i]) or math.isnan(wind_dirs[i]):
            #print('NaN found')
            wind_speeds.pop(i)
            wind_dirs.pop(i)
    print(len(wind_speeds))
    print(len(wind_dirs))

    # convert to knots
    wind_speeds = [s/1.852 for s in wind_speeds]

    speeds_low = []
    speeds_n = []
    speeds_ne = []
    speeds_e = []
    speeds_se = []
    speeds_s = []
    speeds_sw = []
    speeds_w = []
    speeds_nw = []

    def direction_around(d, degrees):
        if degrees == 0:
            return d >= 360-22.5 or d < 22.5
        return d >= degrees - 22.5 and d < degrees + 22.5

    total_data_points = len(wind_speeds)

    for i in range(total_data_points):
        s = wind_speeds[i]
        if s < min_speed:
            speeds_low.append(s)
            continue

        d = wind_dirs[i]
        if direction_around(d, 45):
            speeds_ne.append(s)
        elif direction_around(d, 90):
            speeds_e.append(s)
        elif direction_around(d, 135):
            speeds_se.append(s)
        elif direction_around(d, 180):
            speeds_s.append(s)
        elif direction_around(d, 225):
            speeds_sw.append(s)
        elif direction_around(d, 270):
            speeds_w.append(s)
        elif direction_around(d, 315):
            speeds_nw.append(s)
        else:
            speeds_n.append(s)

    def calc_frac(n):
        return float(n)/total_data_points

    def calc_ave(speeds):
        if len(speeds) == 0:
            return 0
        return np.median(speeds)

    def calc_top(speeds):
        if len(speeds) == 0:
            return 0
        return np.percentile(speeds, 99)


    width_factor = 1.0;


    fraction_low = calc_frac(len(speeds_low))
    fraction_n = calc_frac(len(speeds_n))
    fraction_ne = calc_frac(len(speeds_ne))
    fraction_e = calc_frac(len(speeds_e))
    fraction_se = calc_frac(len(speeds_se))
    fraction_s = calc_frac(len(speeds_s))
    fraction_sw = calc_frac(len(speeds_sw))
    fraction_w = calc_frac(len(speeds_w))
    fraction_nw = calc_frac(len(speeds_nw))

    max_fraction = np.max([fraction_n, fraction_ne, fraction_e, fraction_se, fraction_s, fraction_sw, fraction_w, fraction_nw])
    if max_fraction > 0.125:
        width_factor = 0.125/max_fraction

    ave_n = calc_ave(speeds_n)
    ave_ne = calc_ave(speeds_ne)
    ave_e = calc_ave(speeds_e)
    ave_se = calc_ave(speeds_se)
    ave_s = calc_ave(speeds_s)
    ave_sw = calc_ave(speeds_sw)
    ave_w = calc_ave(speeds_w)
    ave_nw = calc_ave(speeds_nw)

    max_ave = np.max([ave_n, ave_ne, ave_e, ave_se, ave_s, ave_sw, ave_w, ave_nw])

    top_n = calc_top(speeds_n)
    top_ne = calc_top(speeds_ne)
    top_e = calc_top(speeds_e)
    top_se = calc_top(speeds_se)
    top_s = calc_top(speeds_s)
    top_sw = calc_top(speeds_sw)
    top_w = calc_top(speeds_w)
    top_nw = calc_top(speeds_nw)

    max_top = np.max([top_n, top_ne, top_e, top_se, top_s, top_sw, top_w, top_nw])


    # draw the plot

    f, (ax1, ax2) = plt.subplots(1,2)

    ax = ax1

    def draw_wedge(distance_from_center, center_angle, fill_fraction, radius, color):
        angle_start = center_angle - 180*fill_fraction
        angle_end = center_angle + 180*fill_fraction
        center = (math.cos(math.radians(center_angle))*distance_from_center, math.sin(math.radians(center_angle))*distance_from_center)
        sector = patches.Wedge(center, radius, angle_start, angle_end, fill=True, facecolor=color)
        ax.add_patch(sector)
        text_center_x = math.cos(math.radians(center_angle))*(distance_from_center+radius) * 1.2
        text_center_y = math.sin(math.radians(center_angle))*(distance_from_center+radius) * 1.2
        ax.text(text_center_x, text_center_y, f'{radius:.1f}', ha='center', va='center', fontsize=10)

    def draw_wedge2(center_angle, fill_fraction, radius, color):
        if radius <= min_speed:
            return
        angle_start = center_angle - 180*fill_fraction
        angle_end = center_angle + 180*fill_fraction
        center = (0,0)
        sector = patches.Wedge(center, radius, angle_start, angle_end, fill=True, facecolor=color)
        ax.add_patch(sector)
        text_center_x = math.cos(math.radians(center_angle))* radius * 1.2
        text_center_y = math.sin(math.radians(center_angle))* radius * 1.2
        ax.text(text_center_x, text_center_y, f'{radius:.1f}', ha='center', va='center', fontsize=10)

    """    
    draw_wedge(fraction_low * 10, 90, fraction_n, ave_n, 'red')
    draw_wedge(fraction_low * 10, 45, fraction_ne, ave_ne, 'orange')
    draw_wedge(fraction_low * 10, 0, fraction_e, ave_e, 'yellow')
    draw_wedge(fraction_low * 10, -45, fraction_se, ave_se, 'greenyellow')
    draw_wedge(fraction_low * 10, -90, fraction_s, ave_s, 'green')
    draw_wedge(fraction_low * 10, -135, fraction_sw, ave_sw, 'turquoise')
    draw_wedge(fraction_low * 10, 180, fraction_w, ave_w, 'blue')
    draw_wedge(fraction_low * 10, 135, fraction_nw, ave_nw, 'blueviolet')
    """

    draw_wedge2(90, fraction_n*width_factor, ave_n, 'red')
    draw_wedge2(45, fraction_ne*width_factor, ave_ne, 'orange')
    draw_wedge2(0, fraction_e*width_factor, ave_e, 'yellow')
    draw_wedge2(-45, fraction_se*width_factor, ave_se, 'greenyellow')
    draw_wedge2(-90, fraction_s*width_factor, ave_s, 'green')
    draw_wedge2(-135, fraction_sw*width_factor, ave_sw, 'turquoise')
    draw_wedge2(180, fraction_w*width_factor, ave_w, 'blue')
    draw_wedge2(135, fraction_nw*width_factor, ave_nw, 'blueviolet')

    center_circle = patches.Circle((0,0), radius=min_speed, fc = 'azure')
    ax.add_patch(center_circle)
    ax.text(0,0,f'{fraction_low*100:.0f}%', ha='center', va='center', fontsize=8)



    ax.set_xlim(-max_ave*1.25,max_ave*1.25)
    ax.set_ylim(-max_ave*1.25,max_ave*1.25)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Median speeds')

    ax = ax2

    draw_wedge2(90, fraction_n*width_factor, top_n, 'red')
    draw_wedge2(45, fraction_ne*width_factor, top_ne, 'orange')
    draw_wedge2(0, fraction_e*width_factor, top_e, 'yellow')
    draw_wedge2(-45, fraction_se*width_factor, top_se, 'greenyellow')
    draw_wedge2(-90, fraction_s*width_factor, top_s, 'green')
    draw_wedge2(-135, fraction_sw*width_factor, top_sw, 'turquoise')
    draw_wedge2(180, fraction_w*width_factor, top_w, 'blue')
    draw_wedge2(135, fraction_nw*width_factor, top_nw, 'blueviolet')

    center_circle = patches.Circle((0,0), radius=min_speed, fc = 'azure')
    ax.add_patch(center_circle)
    ax.text(0,0,f'{fraction_low*100:.0f}%', ha='center', va='center', fontsize=10)



    ax.set_xlim(-max_top*1.25,max_top*1.25)
    ax.set_ylim(-max_top*1.25,max_top*1.25)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Top speeds')

    print('Low temp', np.percentile(temps, 5))
    print('High temp', np.percentile(temps, 95))
    print(f'Precipitations {float(len([p for p in precip if p > 0]))/len(precip)*100:.1f}%')

    months = [
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    plt.text(0.5,0,
             f'Low temp: {np.percentile(temps, 5):.1f}°C, High temp: {np.percentile(temps, 95):.1f}°C, Precipitations {float(len([p for p in precip if p > 0]))/len(precip)*100:.1f}%, (Cutoff windspeed: {min_speed} kn)',
             ha='center',
             va='bottom',
             transform=f.transFigure)
    plt.text(0.5,0.9,
             f'Week of {months[desired_date.month-1]} {desired_date.day}, Loc: {location._lat:.2f}, {location._lon:.2f}',
             ha='center',
             va='bottom',
             transform=f.transFigure)

if __name__ == "__main__":
    plot_winds(coords, desired_date)
    plt.show()
