#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:57:21 2022

@author: noahbrauer
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py 

#Read in the data

file = '2A.GPM.DPR.V8-20180723.20201028-S075448-E092720.037875.V06A.HDF5'

data = h5py.File(file)

lat = data['NS']['Latitude'][:]
lon = data['NS']['Longitude'][:]

#z_corrected = data['NS']['SLV']['zFactorCorrectedNearSurface'][:]


storm_lat = 25.1398
storm_lon = -91.7397


#Now we need to compute annuli based on distance from the storm center in ten km intervals:
    

def distance(x1,y1,x2,y2):

    dist = np.sqrt(((x2-x1)**2)+(y2-y1)**2) #Returns distance in degrees
    dist_km = dist*111

    return dist, dist_km




#Reshape DPR lat/lon arrays to 1D space:
    
I,J = lat.shape

latitude = lat.reshape(I*J, order = 'F')
longitude = lon.reshape(I*J, order = 'F')    
#ku = z_corrected.reshape(I*J, order = 'F')
#ku[ku<10] = np.nan


#Compute distance from center for all DPR lat/lon points

distance_from_center = []

for i in range(len(latitude)):
    
    
    storm_distance = distance(storm_lon,storm_lat,longitude[i],latitude[i])
    distance_from_center.append(storm_distance)


'''

dist = np.arange(10,200,step = 10)


fontsize = 20
plt.plot(dist,kupr)
plt.xlabel('Distance From Center (km)', size = fontsize)
plt.ylabel('KuPR (dBZ)', size = fontsize)
plt.title('Annuli-Averaged GPM DPR KuPR (Hurricane Zeta 10/28/2020)', size = fontsize)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.show()

'''

#%%


#Now let's integrate the algorithm to compute slope of KuPR in the liquid phase:
#Read in Ku-data 

ku_vert = data['NS']['SLV']['zFactorCorrected'][:]
freezing = data['NS']['VER']['heightZeroDeg'][:]


#Reshape Ku and freezing-level data


I,J,K = ku_vert.shape
ku_reshape = ku_vert.reshape(I*J,K, order = 'F')
freezing_reshape = freezing.reshape(I*J, order = 'F')/1000
latitude = lat.reshape(I*J, order = 'F')
longitude = lon.reshape(I*J, order = 'F')

        
ku_reshape = ku_reshape[:,::-1]

x_bins = np.arange(10,60,1)
y_bins = np.arange(0,21.988032,0.124932)


#Normalize height w.r.t 0degC isotherm

norm_alt =[]

for i in range(len(freezing_reshape)):
    
    norm_altitude = y_bins - freezing_reshape[i]
    norm_alt.append(norm_altitude)

#Now do all the masking in order to perform linear regression

norm_alt = np.asarray(norm_alt)


norm_altitude = norm_alt.copy()
#norm_altitude[(norm_altitude>-1)|(norm_altitude<-3)] = -9999
#norm_altitude = np.ma.masked_where(norm_altitude<-10,norm_altitude)
#norm_altitude = np.ma.masked_where(norm_altitude<-3,norm_altitude)


ku_reshaped = ku_reshape.copy()
#ku_reshaped[(norm_altitude>-1)|(norm_altitude<-3)] = -9999
ku_reshaped[(norm_altitude>-1)|(norm_altitude<-3)] = np.nan
#ku_reshaped = np.ma.masked_where(ku_reshaped<0,ku_reshaped)





#Now compute the linear regression
#The bug is in here:
from scipy import stats
from sklearn.linear_model import LinearRegression

slope_warm = []

for i in range(ku_reshaped.shape[0]):
    
    
    
    if len(ku_reshaped[i][~np.isnan(ku_reshaped[i])]) == 0:
        slope_warm.append(np.nan)
        continue

    #Slice all points of ku_reshaped to remove any values l
    
    sliced_norm_altitude = norm_altitude[i][(norm_altitude[i] <= -1) & (norm_altitude[i] >= -3)]
    sliced_ku_reshaped = ku_reshaped[i][(norm_altitude[i] <= -1) & (norm_altitude[i] >= -3)]
    
    filtered_norm_altitude = sliced_norm_altitude[~np.isnan(sliced_ku_reshaped)]
    filtered_ku_reshaped = sliced_ku_reshaped[~np.isnan(sliced_ku_reshaped)]
    
    if len(filtered_ku_reshaped) <= 5:
        slope_warm.append(np.nan)
        continue
        
    
    slope = stats.linregress(filtered_ku_reshaped,filtered_norm_altitude)[0]
    #print(slope)
    #model = LinearRegression().fit(ku_reshaped[i,:],norm_altitude[i,:])
    #slope = model.score(ku_reshaped[i,:],norm_altitude[i,:])
    slope_warm.append(slope)
    
   
    
   

slope_warm = np.asarray(slope_warm)

slope_liquid = slope_warm.reshape(I,J, order = 'F')

slope_liquid[(slope_liquid<-2)| (slope_liquid>2)] = np.nan


#%%


idxs = np.where(~ku_reshaped.mask)[0]
for idx in idxs:
    print(f"index: {idx}, length = {len(ku_reshaped[idx,:])}")




#%%
#Let's plot on a grid now

from mpl_toolkits.basemap import Basemap

cmin = -1.25; cmax = 1.25; cint = 0.1; clevs = np.round(np.arange(cmin,cmax,cint),2)
nlevs = len(clevs) - 1; cmap = plt.get_cmap(name='turbo',lut=nlevs)

plt.figure(figsize=(14,14))


xlim = np.array([-94.5,-89]); ylim = np.array([22,28])


label_size = 22


m = Basemap(projection='cyl',lon_0=np.mean(xlim),lat_0=np.mean(ylim),llcrnrlat=ylim[0],urcrnrlat=ylim[1],llcrnrlon=xlim[0],urcrnrlon=xlim[1],resolution='i')
m.drawcoastlines(); m.drawstates(), m.drawcountries()
cs = m.contourf(lon,lat,slope_liquid,clevs,cmap='turbo',extend='both')
cs2 = m.plot(lon[:,0]+0.03,lat[:,0]-0.03,'--k',zorder=4)
cs3 = m.plot(lon[:,-1]-0.03,lat[:,-1]+0.03,'--k',zorder=4)
parallels = np.arange(-90,90,step = 2)
m.drawparallels(parallels, labels = [True, False, False, False], fontsize = 20)

meridians = np.arange(0,360, step = 2)
m.drawmeridians(meridians, labels = [False, False, False, True], fontsize = 20)

cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize = label_size)
cbar.set_label(label = '[dBZ/km]',size = label_size)
plt.title('Vertical Slope of KuPR in Liquid Phase 10/28/2020 0827 UTC', size = 24)





