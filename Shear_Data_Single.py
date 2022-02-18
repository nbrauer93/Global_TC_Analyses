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
freezing = DPR['NS']['VER']['heightZeroDeg'][:]

ku = DPR['NS']['SLV']['zFactorCorrected'][:]
ku[ku<=12] = np.nan


storm_lat = 13.965
storm_lon = 140.506


#Now add in shear vector

wind_dir = 45 #In degrees 

#shear_dir = np.deg2rad(wind_dir) #Convert from degrees to radians


#Let's create a function to determine the end point of the arrow based on shear direction:

def arrow_endpoint(wind_dir):
    
    if 0<wind_dir<90: #NE shear
        
        angle_deg = 90 - wind_dir
        angle = np.deg2rad(angle_deg)
        dx = -1*np.cos(angle)
        dy = -1*np.sin(angle)
        
    if 90<wind_dir<180: #SE shear
        
        angle_deg = 180 - wind_dir
        angle = np.deg2rad(angle_deg)
        dx = -1*np.cos(angle)
        dy = np.sin(angle)
        
    if 180<wind_dir<270: #SW shear
        
        angle_deg = 270-wind_dir
        angle = np.deg2rad(angle_deg)
        dx = np.cos(angle)
        dy = np.sin(angle)
        
    if 270<wind_dir<360: #NW shear
        
        angle_deg = wind_dir - 270
        angle = np.deg2rad(angle_deg)
        dx = np.cos(angle)
        dy = -1*np.sin(angle)
        
    if wind_dir  == 0 or wind_dir == 360:
        
        dx = 0
        dy = -1
        
    if wind_dir == 90:
        
        dx = -1
        dy = 0
        
    if wind_dir == 180:
        
        dx = 0
        dy = 1
        
    if wind_dir == 270:
        
        dx = 1
        dy = 0
        

    return [dx,dy]


dx = arrow_endpoint(wind_dir)[0]
dy = arrow_endpoint(wind_dir)[1]


#Now compute and plot a perpendicular line 


if dy != 0 :

    perp_slope = (-dx/dy)
    intercept = storm_lat+(dx/dy)*storm_lon
    
else:
    
    perp_slope = 0
    intercept = storm_lon

#Now set up two arbitrary arrays for lat and lon based off storm center:
#Line length refers to desired length of arrow (detertmine by radius of maximum wind?)    

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


#Now partition KuPR (vertical profiles into shear-relative-quadrants)

#Ku has shape of latitude, angle (ray), height
#Let's contrain the latitude dimension to each shear-relative quadrant
#Index for each quadrant.




def shear_quadrants(lon, lat, storm_lon, storm_lat, shear_dir):
    
    if 0<shear_dir<90:
        
        dr_quad = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon))[0]
        dl_quad = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        ul_quad = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon<=lon)&(lon<storm_lon+np.abs(dx)))[0]
        ur_quad = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        
        
    if 90<shear_dir<180:
        
        dr_quad = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        dl_quad = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        ul_quad = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        ur_quad = np.where(storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy)&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        
    
    if 180<shear_dir<270:
        
        dr_quad = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        dl_quad = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        ul_quad = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon))[0]
        ur_quad = np.where((storm_lon-np.abs(dy)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        
        
    if 270<shear_dir<360:
        
        dr_quad = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        dl_quad = np.where(storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy)&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        ul_quad = np.where(storm_lat<=lat)&(lat<=storm_lat+np.abs(dy)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx)))[0]
        ur_quad = np.where(storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon))[0]
        
        
    return dr_quad,dl_quad,ul_quad,ur_quad



#Now index ku into each quadrant:


dr_index = shear_quadrants(lon,lat,storm_lon,storm_lat,wind_dir)[0]     
                           
   
ku_dr_quad = ku[dr_index,:,:]     
zero_isotherm = freezing[dr_index,:]



#Designate a Z-direction
x_bins = np.arange(10,60,1)
y_bins = np.arange(0,21.988032,0.124932)


ku_1 = ku_dr_quad[95,4,:]

       
        
plt.plot(ku_1.T,y_bins[::-1]) 
plt.show()       
  
#ku_dr_quad has shape (461,49,176) or footprint, angle, height        
    



#Normalize w.r.t to the zero degree isotherm
#Altitude - 0deg isotherm 
#Loop through each freezing level height for all footprints and all angles


isotherm_normalized = np.ones((ku_dr_quad.shape))*np.nan


for i in range(ku_dr_quad.shape[0]):
    
    for j in range(ku_dr_quad.shape[1]):
        
        for k in range(ku_dr_quad.shape[2]):
            
            
            isotherm_normalized[i,j,k] = y_bins[k] - zero_isotherm[i,j]/1000


#Loop through and plot all vertical profiles in a designated quadrant 


for i in range(ku_dr_quad.shape[0]):

    for j in range(ku_dr_quad.shape[1]):
        
        
        fig,ax = plt.subplots(figsize=(14,14)) 
        
        
        xlabels = np.arange(10,65,5)
        #ylabels = np.arange(0,13.5,0.5)
        ylabels = np.arange(-8,8,1)

        plt.xticks(xlabels)
        plt.yticks(ylabels)


        plt.xlim(10,55) 
        #plt.ylim(0,12)
        
        ax.set_xticklabels(['10','15','20', '25', '30', '35', '40', '45', '50','55','60'], size = 20)
        plt.yticks(size =20) 
        
        #plt.plot(ku_dr_quad[i,j,:].T, y_bins[::-1])
        #plt.hlines(zero_isotherm[i,j]/1000,np.nanmin(x_bins), np.nanmax(x_bins), colors = 'k', linestyles = 'dashed')
        plt.plot(ku_dr_quad[i,j,:].T, isotherm_normalized[i,j,:][::-1])
        plt.title('KuPR Profiles in Downshear Right Quadrant', fontsize = 26)
        plt.xlabel('[dBZ]', fontsize = 22)
        #plt.ylabel('[km]', fontsize = 20)
        plt.ylabel('Normalized Altitude [km]', fontsize = 22)
        plt.show()

#Now plot on a lat-lon grid

z[z<12] = np.nan


cmin = 12.; cmax = 70.; cint = 2; clevs = np.round(np.arange(cmin,cmax,cint),2)
nlevs = len(clevs) - 1; cmap = plt.get_cmap(name='turbo',lut=nlevs)

plt.figure(figsize=(14,14))
  

xlim = np.array([136,144]); ylim = np.array([10,17])


label_size = 22

  
m = Basemap(projection='cyl',lon_0=np.mean(xlim),lat_0=np.mean(ylim),llcrnrlat=ylim[0],urcrnrlat=ylim[1],llcrnrlon=xlim[0],urcrnrlon=xlim[1],resolution='i')
m.drawcoastlines(); m.drawstates(), m.drawcountries()
cs = m.contourf(lon,lat,z,clevs,cmap='turbo',extend='both')
cs2 = m.plot(lon[:,0]+0.03,lat[:,0]-0.03,'--k',zorder=4)
cs3 = m.plot(lon[:,-1]-0.03,lat[:,-1]+0.03,'--k',zorder=4)
parallels = np.arange(-90,90,step = 2)
m.drawparallels(parallels, labels = [True, False, False, False])

meridians = np.arange(0,360, step = 2)
m.drawmeridians(meridians, labels = [False, False, False, True])

cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize = label_size)
cbar.set_label(label = '[dBZ]',size = label_size)
plt.clim(12,70)
    
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



plt.title('Near-Surface KuPR 20180911-S034010-E051243', size = 28)
plt.show
