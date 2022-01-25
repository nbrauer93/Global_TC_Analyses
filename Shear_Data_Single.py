#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:42:26 2022

@author: noahbrauer
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
from mpl_toolkits.basemap import Basemap
from shapely.geometry import LineString



#From Typhoon 

file = '2A.GPM.DPR.V8-20180723.20180911-S034010-E051243.025770.V06A.HDF5'

DPR = h5py.File(file, 'r')

lat = DPR['NS']['Latitude'][:,:]
lon = DPR['NS']['Longitude'][:,:]
z = DPR['NS']['SLV']['zFactorCorrectedNearSurface'][:] #nscan x nray (7934,49)


storm_lat = 13.965
storm_lon = 140.506




#Now add in shear vector

wind_dir = 255 #In degrees 

shear_dir = np.deg2rad(wind_dir) #Convert from degrees to radians


#Let's create a function to determine the end point of the arrow based on shear direction:

def arrow_endpoint(shear_dir):
    
    if 0<shear_dir<90: #NE shear
        
        dx = -1*np.cos(shear_dir)
        dy = -1*np.sin(shear_dir)
        
    if 90<shear_dir<180: #SE shear
        
        dx = -1*np.cos(shear_dir)
        dy = np.sin(shear_dir)
        
    if 180<shear_dir<270: #SW shear
        
        dx = np.cos(shear_dir)
        dy = np.sin(shear_dir)
        
    if 270<shear_dir<360: #NW shear
        
        dx = -1*np.sin(shear_dir)
        dy = np.cos(shear_dir)
        

    return [dx,dy]


dx = arrow_endpoint(shear_dir)[0]
dy = arrow_endpoint(shear_dir)[1]


#Now compute and plot a perpendicular line 


perp_slope = (-dx/dy)

intercept = storm_lat+(dx/dy)*storm_lon

#Now set up two arbitrary arrays for lat and lon based off storm center:
    
line_length = 0.75    
    
lats = np.array([storm_lat-line_length, storm_lat + line_length])
lons = np.array([storm_lon-line_length, storm_lon + line_length])


#So now to plot line perpendicular to arrow:
# x = lons
# y = (perp_slope)*lons + intercept
    

#Now extend the arrow down past the point
#Arrow length is simply sqrt(dx**2 + dy**2)

arrow_length = np.sqrt((dx**2)+ (dy**2))

coords_lat = np.array([storm_lat,storm_lat - dy])
coords_lon = np.array([storm_lon,storm_lon- dx])



z[z<12] = np.nan


cmin = 12.; cmax = 70.; cint = 2.5; clevs = np.round(np.arange(cmin,cmax,cint),2)
nlevs = len(clevs) - 1; cmap = plt.get_cmap(name='turbo',lut=nlevs)

plt.figure(figsize=(10,10))
  

xlim = np.array([136,144]); ylim = np.array([10,17])



 

  
m = Basemap(projection='cyl',lon_0=np.mean(xlim),lat_0=np.mean(ylim),llcrnrlat=ylim[0],urcrnrlat=ylim[1],llcrnrlon=xlim[0],urcrnrlon=xlim[1],resolution='i')
m.drawcoastlines(); m.drawstates(), m.drawcountries()
cs = m.contourf(lon,lat,z,clevs,cmap='turbo',extend='both')
cs2 = m.plot(lon[:,0]+0.03,lat[:,0]-0.03,'--k',zorder=4)
cs3 = m.plot(lon[:,-1]-0.03,lat[:,-1]+0.03,'--k',zorder=4)
parallels = np.arange(-90,90,step = 2)
m.drawparallels(parallels, labels = [True, False, False, False])

meridians = np.arange(0,360, step = 2)
m.drawmeridians(meridians, labels = [False, False, False, True])

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
    
x2star,y2star = m(storm_lon,storm_lat)
m.plot(x2star,y2star,'r.',markersize=26, color = 'k')

arrow_length_x = dx
arrow_length_y = dy




x,y = m(storm_lon,storm_lat)
x2,y2 = m(dx,dy)
plt.arrow(x,y,x2,y2, width = 0.008, head_width = 0.2, head_length = 0.2, color = 'k')

x3,y3 = lons, (perp_slope)*lons + intercept
m.plot(x3,y3,linewidth = 2, color = 'k')

#Now draw another line extending down from the storm_lon, storm_lat point

x4,y4 = coords_lon,coords_lat
m.plot(x4,y4, linewidth = 2, color = 'k')



plt.title('Near-Surface KuPR 20180911-S034010-E051243', size = 24)
plt.show