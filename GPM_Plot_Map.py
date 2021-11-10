#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:22:43 2021

@author: noahbrauer
"""


import h5py 
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
import glob
from mpl_toolkits.basemap import Basemap



files = glob.glob('*.HDF5')


#Read in file, extract lat, lons, PSD



gpm_lat = np.array([14.5,25.2,-17.69,14,17])
gpm_lon = np.array([176,-112,117.6,131,123])

radius = 10
xlim = np.array([gpm_lon-radius,gpm_lon+radius])  ; ylim = np.array([gpm_lat-radius,gpm_lat+radius]) 
   
xlim_west = xlim[0]
xlim_east = xlim[1]
ylim_south = ylim[0]
ylim_north = ylim[1]
    
lon_mean = []
lat_mean = []
    
for i in range(len(xlim_west)):
    
    lon_mean.append((xlim_west[i]+xlim_east[i])/2)
    lat_mean.append((ylim_north[i]+ylim_south[i])/2)

for file in range(len(files)):
    data = h5py.File(files[file])
    print(file)

    z_corrected = data['NS']['SLV']['zFactorCorrectedNearSurface'][:]
    
    
    
    lat = data['NS']['Latitude'][:]
    lon = data['NS']['Longitude'][:]

#Lat and lon corresponding to the DPR overpass locations




    z_corrected[z_corrected<20] = np.nan
    
    cmin = 12.; cmax = 60.; cint = 2.5; clevs = np.round(np.arange(cmin,cmax,cint),2)
    nlevs = len(clevs) - 1; cmap = plt.get_cmap(name='turbo',lut=nlevs)

    plt.figure(figsize=(20,20))


 



     
        
    m = Basemap(projection='cyl',lon_0=lon_mean[file],lat_0=lat_mean[file],llcrnrlat=ylim_south[file],urcrnrlat=ylim_north[file],llcrnrlon=xlim_west[file],urcrnrlon=xlim_east[file],resolution='l')
    m.drawcoastlines(); m.drawstates(), m.drawcountries()
    
    
    cs = m.contourf(lon,lat,z_corrected,clevs,cmap='turbo',extend='both')
    cs2 = m.plot(lon[:,0]+0.03,lat[:,0]-0.03,'--k',zorder=1)
    cs3 = m.plot(lon[:,-1]-0.03,lat[:,-1]+0.03,'--k',zorder=1)





    cbar = m.colorbar(cs,size='2%')
    cbar.ax.set_ylabel('[dBZ]',name='Calibri',size=18)
    cticks = []
    for i in clevs:
        cticks.append(int(i)) if i.is_integer() else cticks.append(i)
        cbar.set_ticks(clevs[::4])
        cbar.set_ticklabels(cticks[::4])
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_family('Calibri')
        i.set_size(20)

    plt.title('GPM Overpass KuPR', size = 20)
    plt.show()
    


