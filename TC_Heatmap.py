#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:08:25 2021

@author: noahbrauer
"""


#Import all libraries

import matplotlib.pyplot as plt
import numpy as np
import datetime
import re
import sys, os
from pyorbital.orbital import Orbital
from math import sin, cos, sqrt, atan2, radians

from overpasstimeswithlist import distance_km,get_TLE,find_overpasstimes

from mpl_toolkits.basemap import Basemap

import csv

import pandas as pd


#Now open the CSV file



ibtracks_data = pd.read_csv('ibtracs.ALL.list.v04r00.csv')

#Extract time attributes


season = ibtracks_data['SEASON'][1:]
season = season.copy()

season_array = pd.Series.to_numpy(season, dtype = 'float')




season_index = np.where(season_array>2013)[0]


season_GPM = season_array[season_index]


#Extract lats and lons:

lat = ibtracks_data['LAT'][season_index]
lon = ibtracks_data['LON'][season_index]


latitude = pd.Series.to_numpy(lat,dtype = 'float')
longitude = pd.Series.to_numpy(lon,dtype = 'float')


#Extract basin and name

basin = ibtracks_data['BASIN'][season_index]
name = ibtracks_data['NAME'][season_index]


name = name.copy()

#NaN out all the unnamed storms

name[name == "NOT_NAMED"] = np.nan

name = pd.Series.to_numpy(name, dtype = 'str')

#Extract exact times


time = ibtracks_data['ISO_TIME'][season_index]
time_numpy = pd.Series.to_numpy(time, dtype = 'str')

#%%

#Create our polygons for each TC Basin





def draw_vert_line(lat_min,lat_max,longitude):
    
    latitude_vert = np.arange(lat_min,lat_max,step = 1)
    longitude_vert = np.full((lat_max-lat_min),longitude)
    
    return longitude_vert,latitude_vert

def draw_horiz_line(lon_min,lon_max,latitude):
    
    longitude_horiz = np.arange(lon_min,lon_max,step = 1)
    latitude_horiz = np.full((lon_max-lon_min),latitude)
    
    return longitude_horiz,latitude_horiz


def positive_slope(lat_min,lat_max,lon_min,lon_max):
    
    lat = [lat_min,lat_max]
    lon = [lon_min,lon_max]
    
    return lon,lat


def negative_slope(lat_min,lat_max,lon_min,lon_max):
    
    lat = [lat_min,lat_max]
    lon = [lon_max,lon_min]
    
    return lon,lat











#%%

#Let's plot a heatmap on a grid of all TCs 



#Establish a lat-lon grid from -180 to 180, -90 to 90 in one degree intervals


lon_grid = np.arange(-180.,180., step = 1)

lat_grid = np.arange(-90.,90., step = 1)

#Now mesh to a grid

lon2,lat2 = np.meshgrid(lon_grid,lat_grid)

#Round lat and lon to nearest integers

lon_rounded = np.round(longitude)
lat_rounded = np.round(latitude)


#####Now count the frequency of each lat and lon value in both lat and lon arrays######



gridded_frequency = np.histogram2d(lon_rounded,lat_rounded, bins = [lon_grid,lat_grid])[0]


gridded_frequency[gridded_frequency==0] = np.nan

#Now let's plot the gridded frequency

plt.figure(figsize=(14,14))
#m = Basemap(llcrnrlon = -180, llcrnrlat = -50, urcrnrlon = 180, urcrnrlat = 50)
m = Basemap(llcrnrlon =-180 , llcrnrlat = -60, urcrnrlon = 180, urcrnrlat = 60)
m.drawcoastlines()
m.drawstates()
m.drawcountries()

x,y = m(lon2,lat2)


 

cs = m.pcolormesh(x,y,gridded_frequency.T, cmap = 'Reds', vmin = 0, vmax = 20)

cbar = m.colorbar(cs,size='2%')
cbar.ax.set_ylabel('[Frequency]',name='Calibri',size=18)

plt.title('2014-2021 Tropical Cyclones', size = 18)


parallels = np.arange(-70,70, step = 10)
m.drawparallels(parallels, labels = [True, False, False, False])

meridians = np.arange(-180, 180, step = 20)
m.drawmeridians(meridians, labels = [False, False, False, True])

linewidth = 4  
font_size = 17
name = 'Calibri'
    
vertical_pac = draw_vert_line(-40,40,100)
x,y = m(vertical_pac[0],vertical_pac[1])
m.plot(x,y, color = 'k', linewidth = linewidth)   


horiz_pac = draw_horiz_line(105,180,0)
x,y = m(horiz_pac[0],horiz_pac[1])
m.plot(x,y, color = 'k', linewidth = linewidth)

horiz_indian = draw_horiz_line(42,105,0)
x,y = m(horiz_indian[0],horiz_indian[1])
m.plot(x,y,color = 'k', linewidth = linewidth)

horiz_pac = draw_horiz_line(-180,-80,0)
x,y = m(horiz_pac[0],horiz_pac[1])
m.plot(x,y, color = 'k', linewidth = linewidth)

vert_pac = draw_vert_line(0,40,-179)
x,y = m(vert_pac[0],vert_pac[1])
m.plot(x,y,color = 'k', linewidth = linewidth)


lon = negative_slope(10,40,-120,-83)[0]
lat = negative_slope(10,40,-120,-83)[1]
x,y = m(lon,lat)
m.plot(x,y, color = 'k', linewidth = linewidth)


lon = positive_slope(-33,0,28,42)[0]
lat = positive_slope(-33,0,28,42)[1]
x,y = m(lon,lat)
m.plot(x,y,color = 'k', linewidth = linewidth)

vertical_indian = draw_vert_line(0,30,42)
x,y = m(vertical_indian[0],vertical_indian[1])
m.plot(x,y,color = 'k', linewidth = linewidth)


###Add Text
x2star,y2star = m(-155,32)
plt.text(x2star,y2star,'ECPAC' , color = 'k', fontsize = font_size)

x3star,y3star = m(-150,-24)
plt.text(x3star,y3star,'SPAC' , color = 'k', fontsize = font_size)

x4star,y4star = m(-54,26)
plt.text(x4star,y4star,'ATL' , color = 'k', fontsize = font_size)

x5star,y5star = m(57.5,5)
plt.text(x5star,y5star,'NIND' , color = 'k', fontsize = font_size)

x6star,y6star = m(69,-23)
plt.text(x6star,y6star,'SIND' , color = 'k', fontsize = font_size)

x7star,y7star = m(160,-31)
plt.text(x7star,y7star,'SPAC' , color = 'k', fontsize = font_size)

x8star,y8star = m(145,22)
plt.text(x8star,y8star,'NWPAC' , color = 'k', fontsize = font_size)
plt.show()







