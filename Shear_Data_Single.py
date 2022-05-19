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
from scipy import stats



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

wind_dir = 115 #In degrees 

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

#print(dx)
#print(dy)


#Now compute and plot a perpendicular line 


if dy != 0 :

    perp_slope = (-1*dx/dy)
    intercept = storm_lat+(dx/dy)*storm_lon
    
else:
    
    perp_slope = 0
    intercept = storm_lon

#print(intercept)
#print(perp_slope)


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


#%%

#Reshape lat and lon:
    
I,J = lat.shape

lat_reshape = lat.reshape(I*J, order = 'F')
lon_reshape = lon.reshape(I*J,order = 'F')


#Now import the function to partition by distance from storm center

def distance(x1,y1,x2,y2):

    dist = np.sqrt(((x2-x1)**2)+(y2-y1)**2) #Returns distance in degrees
    dist_km = dist*111

    return dist, dist_km

#Now compute distance from storm center:
    
distance_from_center = distance(storm_lon,storm_lat,lon_reshape,lat_reshape)[1]




#ur_quad = np.where(storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy)&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx)))[0]

#%%
#Now partition KuPR (vertical profiles into shear-relative-quadrants)

#Ku has shape of latitude, angle (ray), height
#Let's contrain the latitude dimension to each shear-relative quadrant
#Index for each quadrant.
'''
from geopy.distance import geodesic
geodesic(kilometers=500).destination((start_lat,start_lon), 115)
'''


eyewall_dist = 15
inner_core_range = np.array([15,60])



def shear_quadrants(lon, lat, storm_lon, storm_lat, shear_dir, distance_from_center):
    
    if 0<shear_dir<90:
        
        dr_quad_eyewall = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)-np.abs(dy)<=lon)&(lon<=storm_lon)&(distance_from_center<eyewall_dist))[0]
        dl_quad_eyewall = np.where((storm_lat-np.abs(dy) - np.abs(dx) <=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center<eyewall_dist))[0]
        ul_quad_eyewall = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon<=lon)&(lon<storm_lon+np.abs(dx) + np.abs (dy))&(distance_from_center<eyewall_dist))[0]
        ur_quad_eyewall = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy)+np.abs(dx))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center<eyewall_dist))[0]
        
        dr_quad_inner = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)-np.abs(dy)<=lon)&(lon<=storm_lon)&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        dl_quad_inner = np.where((storm_lat-np.abs(dy) - np.abs(dx) <=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        ul_quad_inner = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon<=lon)&(lon<storm_lon+np.abs(dx) + np.abs (dy))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        ur_quad_inner = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy)+np.abs(dx))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center>15)&(distance_from_center<=60))[0]
        
        
    if 90<shear_dir<180:
        
        dr_quad_eyewall = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy)+np.abs(dx))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center<eyewall_dist))[0]
        dl_quad_eyewall = np.where((storm_lat-np.abs(dx)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)- np.abs(dy)<=lon)&(lon<=storm_lon)&(distance_from_center<eyewall_dist))[0]
        ul_quad_eyewall = np.where((storm_lat-np.abs(dy)- np.abs(dx)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dy)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center<eyewall_dist))[0]
        ur_quad_eyewall = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dx)+ np.abs(dx))&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx) + np.abs(dy))&(distance_from_center<eyewall_dist))[0]
        
        dr_quad_inner = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy)+np.abs(dx))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        dl_quad_inner = np.where((storm_lat-np.abs(dx)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)- np.abs(dy)<=lon)&(lon<=storm_lon)&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        ul_quad_inner = np.where((storm_lat-np.abs(dy)- np.abs(dx)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dy)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        ur_quad_inner = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dx)+ np.abs(dx))&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx) + np.abs(dy))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        
    
    if 180<shear_dir<270:
        #Good
        dr_quad_eyewall = np.where((storm_lat-np.abs(dx)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx)+np.abs(dy))&(distance_from_center<eyewall_dist))[0]
        dl_quad_eyewall = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy)+np.abs(dx))&(storm_lon-np.abs(dx)-np.abs(dy)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center<eyewall_dist))[0]
        ul_quad_eyewall = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy)+ np.abs(dx))&(storm_lon-np.abs(dx)- np.abs(dy)<=lon)&(lon<=storm_lon)&(distance_from_center<eyewall_dist))[0]
        ur_quad_eyewall = np.where((storm_lat-np.abs(dy) - np.abs(dx)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center<eyewall_dist))[0]
        
        dr_quad_inner = np.where((storm_lat-np.abs(dx)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx)+np.abs(dy))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        dl_quad_inner = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy)+np.abs(dx))&(storm_lon-np.abs(dx)-np.abs(dy)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        ul_quad_inner = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dy)+ np.abs(dx))&(storm_lon-np.abs(dx)- np.abs(dy)<=lon)&(lon<=storm_lon)&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        ur_quad_inner = np.where((storm_lat-np.abs(dy) - np.abs(dx)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        
        
    if 270<shear_dir<360:
        #Good
        dr_quad_eyewall = np.where((storm_lat-np.abs(dy)- np.abs(dx)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center<eyewall_dist))[0]
        dl_quad_eyewall = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dx))&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center<eyewall_dist))[0]
        ul_quad_eyewall = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy)+ np.abs(dx))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center<eyewall_dist))[0]
        ur_quad_eyewall = np.where((storm_lat-np.abs(dx)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)- np.abs(dy)<=lon)&(lon<=storm_lon)&(distance_from_center<eyewall_dist))[0]
        
        dr_quad_inner = np.where((storm_lat-np.abs(dy)- np.abs(dx)<=lat)&(lat<=storm_lat)&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        dl_quad_inner = np.where((storm_lat-np.abs(dy)<=lat)&(lat<=storm_lat+np.abs(dx))&(storm_lon<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        ul_quad_inner = np.where((storm_lat<=lat)&(lat<=storm_lat+np.abs(dy)+ np.abs(dx))&(storm_lon-np.abs(dx)<=lon)&(lon<=storm_lon+np.abs(dx))&(distance_from_center>inner_core_range[0])&(distance_from_center<=[1]))[0]
        ur_quad_inner = np.where((storm_lat-np.abs(dx)<=lat)&(lat<=storm_lat+np.abs(dy))&(storm_lon-np.abs(dx)- np.abs(dy)<=lon)&(lon<=storm_lon)&(distance_from_center>inner_core_range[0])&(distance_from_center<=inner_core_range[1]))[0]
        
        
    return dr_quad_eyewall,dl_quad_eyewall,ul_quad_eyewall,ur_quad_eyewall,dr_quad_inner,dl_quad_inner,ul_quad_inner,ur_quad_inner



#Now index ku into each quadrant:


dr_index_eyewall = shear_quadrants(lon_reshape,lat_reshape,storm_lon,storm_lat,wind_dir,distance_from_center)[0]     
dl_index_eyewall = shear_quadrants(lon_reshape,lat_reshape,storm_lon,storm_lat,wind_dir,distance_from_center)[1]
ul_index_eyewall = shear_quadrants(lon_reshape,lat_reshape,storm_lon,storm_lat,wind_dir,distance_from_center)[2]
ur_index_eyewall = shear_quadrants(lon_reshape,lat_reshape,storm_lon,storm_lat,wind_dir,distance_from_center)[3]       

dr_index_inner = shear_quadrants(lon_reshape,lat_reshape,storm_lon,storm_lat,wind_dir,distance_from_center)[4]     
dl_index_inner = shear_quadrants(lon_reshape,lat_reshape,storm_lon,storm_lat,wind_dir,distance_from_center)[5]
ul_index_inner = shear_quadrants(lon_reshape,lat_reshape,storm_lon,storm_lat,wind_dir,distance_from_center)[6]
ur_index_inner = shear_quadrants(lon_reshape,lat_reshape,storm_lon,storm_lat,wind_dir,distance_from_center)[7]                              
                          
 
    
 
 

#Designate a Z-direction
x_bins = np.arange(10,60,1)
y_bins = np.arange(0,21.988032,0.124932)




       
        
#plt.plot(ku_1.T,y_bins[::-1]) 
#plt.show()       
 
#ku_dr_quad has shape (461,49,176) or footprint, angle, height        
    







#Reshape KuPR to be space x height


I,J,K = ku.shape

ku_reshape = ku.reshape(I*J,K, order = 'F')
freezing_reshape = freezing.reshape(I*J,order = 'F')/1000 # Convert to meters


#Normalize w.r.t to the zero degree isotherm
#Altitude - 0deg isotherm 
#Loop through each freezing level height for all footprints and all angles

freezing_norm = []


for i in range(len(freezing_reshape)):
    
    freezing_normalized = y_bins - freezing_reshape[i]
    freezing_norm.append(freezing_normalized)


freezing_norm = np.asarray(freezing_norm)



#Now index for each shear relative quadrant and annulus:
    
    
ku_dl_eyewall = ku_reshape[dl_index_eyewall,::-1]
ku_dr_eyewall = ku_reshape[dr_index_eyewall,::-1]
ku_ul_eyewall = ku_reshape[ul_index_eyewall,::-1]
ku_ur_eyewall = ku_reshape[ur_index_eyewall,::-1]

ku_dl_inner = ku_reshape[dl_index_inner,::-1]
ku_dr_inner = ku_reshape[dr_index_inner,::-1]
ku_ul_inner = ku_reshape[ul_index_inner,::-1]
ku_ur_inner = ku_reshape[ur_index_inner,::-1]



freezing_dl_eyewall = freezing_norm[dl_index_eyewall]
freezing_dr_eyewall = freezing_norm[dr_index_eyewall]
freezing_ul_eyewall = freezing_norm[ul_index_eyewall]
freezing_ur_eyewall = freezing_norm[ur_index_eyewall]    

freezing_dl_inner = freezing_norm[dl_index_inner]
freezing_dr_inner = freezing_norm[dr_index_inner]
freezing_ul_inner = freezing_norm[ul_index_inner]
freezing_ur_inner = freezing_norm[ur_index_inner]    




#Now we need to compute slope both within the warm cloud layer and the ice-phase layer
#Mask everything greater than normalized altitude of -1 km abd everything less than -3 km

ku_dl_eyewall = ku_dl_eyewall.copy()
ku_dl_eyewall[(freezing_dl_eyewall>-1) | (freezing_dl_eyewall<-3)] = -9999
ku_dl_eyewall = np.ma.masked_where(ku_dl_eyewall<0,ku_dl_eyewall)

ku_dr_eyewall = ku_dr_eyewall.copy()
ku_dr_eyewall[(freezing_dr_eyewall>-1) | (freezing_dr_eyewall<-3)] = -9999
ku_dr_eyewall = np.ma.masked_where(ku_dr_eyewall<0,ku_dr_eyewall)

ku_ul_eyewall = ku_ul_eyewall.copy()
ku_ul_eyewall[(freezing_ul_eyewall>-1) | (freezing_ul_eyewall<-3)] = -9999
ku_ul_eyewall = np.ma.masked_where(ku_ul_eyewall<0,ku_ul_eyewall)

ku_ur_eyewall = ku_ur_eyewall.copy()
ku_ur_eyewall[(freezing_ur_eyewall>-1) | (freezing_ur_eyewall<-3)] = -9999
ku_ur_eyewall = np.ma.masked_where(ku_ur_eyewall<0,ku_ur_eyewall)


ku_dl_inner = ku_dl_inner.copy()
ku_dl_inner[(freezing_dl_inner>-1) | (freezing_dl_inner<-3)] = -9999
ku_dl_inner = np.ma.masked_where(ku_dl_inner<0,ku_dl_inner)

ku_dr_inner = ku_dr_inner.copy()
ku_dr_inner[(freezing_dr_inner>-1) | (freezing_dr_inner<-3)] = -9999
ku_dr_inner = np.ma.masked_where(ku_dr_inner<0,ku_dr_inner)

ku_ul_inner = ku_ul_inner.copy()
ku_ul_inner[(freezing_ul_inner>-1) | (freezing_ul_inner<-3)] = -9999
ku_ul_inner = np.ma.masked_where(ku_ul_inner<0,ku_ul_inner)

ku_ur_inner = ku_ur_inner.copy()
ku_ur_inner[(freezing_ur_inner>-1) | (freezing_ur_inner<-3)] = -9999
ku_ur_inner = np.ma.masked_where(ku_ur_inner<0,ku_ur_inner)








            

#%%
'''

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
'''
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
parallels = np.arange(-90,90,step = 0.5)
m.drawparallels(parallels, labels = [True, False, False, False])

meridians = np.arange(0,360, step = 0.5)
m.drawmeridians(meridians, labels = [False, False, False, True])

cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize = label_size)
cbar.set_label(label = '[dBZ]',size = label_size)
plt.clim(12,70)
    
x2star,y2star = m(storm_lon,storm_lat)
#m.plot(x2star,y2star,markersize=26, color = 'k')
m.scatter(x2star,y2star,s=200, color = 'k')


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


#UL bound lon+dx,lat+dy
x5,y5 = m(141.83,13.542)
m.scatter(x5,y5,s=200, color = 'k') #Far northern and western part of domain (tip of arrow w/ shear_dir 115 degrees)

# UR bound lon - dx, lat+dy #Far northern and eastern part of domain
#x6,y6 = m(140.92,15.29)
#m.scatter(x6,y6,s=200, color = 'm')


'''
#UL bound lon+dx,lat+dy
x5,y5 = m(140.083,14.87)
m.scatter(x5,y5,s=200, color = 'k')

# UR bound lon - dx, lat+dy
x6,y6 = m(140.92,14.87)
m.scatter(x6,y6,s=200, color = 'm')

'''



plt.title('Near-Surface KuPR 20180911-S034010-E051243', size = 28)
plt.show

#%%

'''
#Now let's compute all slopes of KuPR within the warm cloud layer 
#Pseudocode:
#We want to end up with a footprint x angle array --> Inputing arrays with dimensions footprint x angle x height
#Let's loop through the each element of the KuPR array and determine indices that are within the vertical layer for all footprints and angles:

#Index for normalized altitude regions less than -1 but greater than -3


ku_warm = ku_dr_quad.copy()
isotherm_normalized_warm = isotherm_normalized.copy()

#ku_warm[(isotherm_normalized_warm>-1)|(isotherm_normalized_warm<-3)] = np.nan
#isotherm_normalized_warm[(isotherm_normalized_warm>-1)|(isotherm_normalized_warm<-3)] = np.nan

 
#iso_test = isotherm_normalized_warm[~np.isnan(isotherm_normalized_warm)]


I,J,K = ku_warm.shape
ku_reshape = ku_warm.reshape(I*J,K, order = 'F')  

iso_reshape = isotherm_normalized_warm.reshape(I*J,K, order = 'F')  

#Loop through each point on horizontal plane for all heights:
    
    
space_index = []    
    
for i in range(iso_reshape.shape[0]):
    
    warm_index = np.where((iso_reshape[~np.isnan(iso_reshape)]))
    
    space_index.append(warm_index)
    
    
'''
    




   
            
        
            
            
            



