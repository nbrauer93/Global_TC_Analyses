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
lon = nc.variables['longitude'][:]
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

#Now set up plotting


title_name = 'DJF Mean Total Integrated Water Vapor'
time = '2014-2020'
title_font_size = 22


cmin = 0; cmax = 60; cint = 2.5; clevs = np.round(np.arange(cmin,cmax,cint),2)

plt.figure(figsize=(20,10))


xlim = np.array([-180,180]); ylim = np.array([-50,50])

m = Basemap(projection='cyl',lon_0=np.mean(xlim),lat_0=np.mean(ylim),llcrnrlat=ylim[0],urcrnrlat=ylim[1],llcrnrlon=xlim[0],urcrnrlon=xlim[1],resolution='l')
m.drawcoastlines(); m.drawstates(); m.drawcountries()

parallels = np.arange(-50,50, step = 10)
m.drawparallels(parallels, labels = [True, False, False, False])

meridians = np.arange(-180, 180, step = 20)
m.drawmeridians(meridians[::2], labels = [False, False, False, True])
cs = m.contourf(lon2,lat2,djf_mean.T, clevs, cmap = 'bwr', extend = 'both')


cbar = m.colorbar(cs,size='2%')
cbar.ax.set_ylabel(r'[$kg$ $m^{-2}$]',size=title_font_size)
plt.title(str(title_name) + str(time),name='Calibri',size=title_font_size)
plot = plt.show(block=False)
