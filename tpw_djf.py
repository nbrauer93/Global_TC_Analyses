#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:29:21 2021

@author: noahbrauer
"""

import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import netCDF4 as nc
import numpy as np
from datetime import datetime, date
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.basemap import Basemap

#Read in file
file = 'tpw_djf.nc'
nc = Dataset(file, 'r')

#Read in data
lat = nc.variables['latitude'][:]
lon = nc.variables['longitude'][:]-180
time = nc.variables['time'][:]
tpw_djf = nc.variables['tcwv'][:]


#Create lat-lon grid:

lat2,lon2 = np.meshgrid(lat, lon)








#Now compute mean across the time axis for each season:

djf_mean = np.nanmean(tpw_djf, axis = 0)
#mam_mean = np.nanmean(tpw_mam, axis = 0)
#jja_mean = np.nanmean(tpw_jja, axis = 0)
#son_mean = np.nanmean(tpw_son, axis = 0)


print(np.nanmax(djf_mean))


djf_mean[djf_mean==0] = np.nan


#%%

#Polygons

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
#Now set up plotting


title_name = 'DJF Mean Total Integrated Water Vapor '
time = '2014-2020'
title_font_size = 22
font_size = 14


cmin = 0; cmax = 65; cint = 2.5; clevs = np.round(np.arange(cmin,cmax,cint),2)

plt.figure(figsize=(20,10))


xlim = np.array([-180,180]); ylim = np.array([-50,50])

m = Basemap(projection='cyl',lon_0=np.mean(xlim),lat_0=np.mean(ylim),llcrnrlat=ylim[0],urcrnrlat=ylim[1],llcrnrlon=xlim[0],urcrnrlon=xlim[1],resolution='l')
m.drawcoastlines(); m.drawstates(); m.drawcountries()

parallels = np.arange(-50,60, step = 10)
m.drawparallels(parallels, labels = [True, False, False, False],size = font_size)

meridians = np.arange(-180, 180, step = 20)
m.drawmeridians(meridians[::2], labels = [False, False, False, True],size = font_size)
cs = m.contourf(lon2.T,lat2.T,djf_mean, clevs, cmap = 'YlGn', extend = 'both')

cbar = plt.colorbar(fraction = 0.02)
cbar.ax.tick_params(labelsize = font_size)
cbar.set_label(label = r'[$kg$ $m^{-2}$]',size = font_size)


plt.title(str(title_name) + str(time),name='Calibri',size=title_font_size)



linewidth = 4  
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
plt.text(x2star,y2star,'ECPAC' , color = 'k', fontsize = font_size, weight = 'bold')

x3star,y3star = m(-150,-24)
plt.text(x3star,y3star,'SPAC' , color = 'k', fontsize = font_size,weight = 'bold')

x4star,y4star = m(-54,26)
plt.text(x4star,y4star,'ATL' , color = 'k', fontsize = font_size,weight = 'bold')

x5star,y5star = m(57.5,5)
plt.text(x5star,y5star,'NIND' , color = 'k', fontsize = font_size,weight = 'bold')

x6star,y6star = m(69,-23)
plt.text(x6star,y6star,'SIND' , color = 'k', fontsize = font_size,weight = 'bold')

x7star,y7star = m(160,-31)
plt.text(x7star,y7star,'SPAC' , color = 'k', fontsize = font_size,weight = 'bold')

x8star,y8star = m(145,22)
plt.text(x8star,y8star,'NWPAC' , color = 'k', fontsize = font_size,weight = 'bold')



plot = plt.show(block=False)
