#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 21:19:38 2022

@author: noahbrauer
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py 
import pandas as pd
import glob
import h5py
import scipy
from scipy import stats
from scipy.stats import norm
import seaborn as sns

file = 'output_no_shear.csv'

#Read in file

data = pd.read_csv(file, usecols = ['storm_date', 'storm_lat', 'storm_lon', 'storm_name','gpm_lat','gpm_lon'])

#Extract IBtracs time, lat, lon for each storm

storm_time = data['storm_date'][:]
storm_lat = data['storm_lat'][:]
storm_lon = data['storm_lon'][:]
storm_name = data['storm_name'][:]
sat_lat = data['gpm_lat'][:]
sat_lon = data['gpm_lon'][:]



#Convert to an array

time = pd.Series.to_numpy(storm_time)
lat = pd.Series.to_numpy(storm_lat)
lon = pd.Series.to_numpy(storm_lon)
name = pd.Series.to_numpy(storm_name)
gpm_lat = pd.Series.to_numpy(storm_lat)
gpm_lon = pd.Series.to_numpy(sat_lon)


#Now designate regions based off lat-

atl_index = np.where((lon>-98)&(lon<0))[0]
epac_index = np.where((lat>0)&(lon>=-180)&(lon<-80))[0]
spac_index = np.where((lat<0)&(lon>100)|(lon<-80))[0]
nwpac_index = np.where((lat>0)&(lon>=100))[0]
nind_index = np.where((lat>0)&(lon>40)&(lon<100))[0]
sind_index = np.where((lat<0)&(lon<100)&(lon>20))[0]



atl_storm = name[atl_index]
epac_storm = name[epac_index]
spac_storm = name[spac_index]
nwpac_storm = name[nwpac_index]
nind_storm = name[nind_index]
sind_storm = name[sind_index]



atl_date = time[atl_index]
epac_date = time[epac_index]
spac_date = time[spac_index]
nwpac_date = time[nwpac_index]
nind_date = time[nind_index]
sind_date = time[sind_index]


atl = np.vstack((atl_storm,atl_date)).T
epac = np.vstack((epac_storm,epac_date)).T
spac = np.vstack((spac_storm,spac_date)).T
nwpac = np.vstack((nwpac_storm,nwpac_date)).T
nind = np.vstack((nind_storm,nind_date)).T
sind = np.vstack((sind_storm,sind_date)).T

#Define function to compute distance:
    
    
def distance(x1,y1,x2,y2):

    dist = np.sqrt(((x2-x1)**2)+(y2-y1)**2) #Returns distance in degrees
    dist_km = dist*111

    return dist, dist_km



#Let's try with a single file


#file = '2A.GPM.DPR.V8-20180723.20140416-S081135-E094408.000742.V06A.HDF5'
#file = '2A.GPM.DPR.V8-20180723.20140711-S173612-E190844.002086.V06A.HDF5'
import sys
file = sys.argv[1]
file_index = int(sys.argv[2])

'''
print("Storm date:")
print(storm_time[file_index])
print("file name")
print(file)
'''

#Open file

DPR = h5py.File(file, 'r')

latitude = DPR['NS']['Latitude'][:]
longitude = DPR['NS']['Longitude'][:]
z = DPR['NS']['SLV']['zFactorCorrectedNearSurface'][:] #nscan x nray (7934,49)
ku = DPR['NS']['SLV']['zFactorCorrected'][:]
ku_uncorrected = DPR['NS']['PRE']['zFactorMeasured'][:]
freezing = DPR['NS']['VER']['heightZeroDeg'][:]

#Read in the latitude for this case on 4/16/2014

case_lat = storm_lat[file_index]
case_lon = storm_lon[file_index]
lat_gpm = gpm_lat[file_index]
lon_gpm = gpm_lon[file_index] 

I,J = latitude.shape
latitude_reshape = latitude.reshape(I*J, order = 'F')

longitude_reshape = longitude.reshape(I*J, order = 'F')



#print(latitude_reshape.shape)

#distance_from_center = distance(case_lon[file_index],case_lat[file_index],longitude_reshape,latitude_reshape)[1]


distance_from_center = []

for i in range(len(longitude_reshape)):
    
    distance_comp = distance(case_lon,case_lat,longitude_reshape[i],latitude_reshape[i])[1]
    distance_from_center.append(distance_comp)
 
#print(len(distance_from_center))


distance_from_center = np.asarray(distance_from_center)

#print(np.nanmax(distance_from_center))
#print(np.nanmin(distance_from_center))


    
    
'''
print("distance from center")
print(distance_from_center)
print(np.nanmax(distance_from_center))
print(np.nanmin(distance_from_center))
'''

#Now let's create annulus around each storm center:
    
    
def annulus(distance_from_center):
    
    r'''
     Define annuli and 15, 60, 110, and 200 km from the storm center:
    '''    
    
    eyewall = np.where(distance_from_center<=15)[0]
    inner_core = np.where((distance_from_center>15)&(distance_from_center<=60))[0]
    outer_core = np.where((distance_from_center>60)&(distance_from_center<=110))[0]
    outer_rainbands = np.where((distance_from_center>110)&(distance_from_center<=200))[0]
    
    return[eyewall,inner_core,outer_core,outer_rainbands]



eyewall = annulus(distance_from_center)[0]
inner_core = annulus(distance_from_center)[1]
outer_core = annulus(distance_from_center)[2]
outer_bands = annulus(distance_from_center)[3]
print("-----------------")



I,J,K = ku.shape

ku_reshape = ku.reshape(I*J,K, order = 'F')
uncorr_reshape = ku_uncorrected.reshape(I*J,K, order = 'F')
freezing_reshape = freezing.reshape(I*J, order = 'F')

ku_reshape[ku_reshape<10] = np.nan
uncorr_reshape[uncorr_reshape<10] = np.nan


ku_eyewall = ku_reshape[eyewall,::-1]
ku_inner = ku_reshape[inner_core,::-1]
ku_outer = ku_reshape[outer_core,::-1]
ku_bands = ku_reshape[outer_bands,::-1]

print(np.nanmax(ku_eyewall))
print(np.nanmax(ku_inner))
print(np.nanmax(ku_outer))

#print(ku_eyewall)
#print(ku_inner)

'''
print("-------------------------")
print("outer_bands:")
print(outer_bands)
print("ku's shape:")
print(ku.shape)
print("ku_reshape shape:")
print(ku_reshape.shape)
print("ku_bands shape:")
print(ku_bands.shape)
'''


uncorr_eyewall = uncorr_reshape[eyewall,::-1]
uncorr_inner = uncorr_reshape[inner_core,::-1]
uncorr_outer = uncorr_reshape[outer_core,::-1]
uncorr_bands = uncorr_reshape[outer_bands,::-1]



freezing_eyewall = freezing_reshape[eyewall]/1000
freezing_inner = freezing_reshape[inner_core]/1000
freezing_outer = freezing_reshape[outer_core]/1000
freezing_bands = freezing_reshape[outer_bands]/1000

#print(freezing_bands.shape)
#print(ku_bands.shape)



#Now we need to extract the height vairables from the DPR files:
    
x_bins = np.arange(10,60,1)
y_bins = np.arange(0,21.988032,0.124932)
#print(len(y_bins))


#Now let's normalize W.R.T the 0 degC isotherm: altitude - zero degree C isotherm



norm_eyewall = np.ones((ku_eyewall.shape))*np.nan

for i in range(len(freezing_eyewall)):
    
    norm_eyewall[i,:] = y_bins - freezing_eyewall[i]


norm_inner = np.ones((ku_inner.shape))*np.nan

for i in range(len(freezing_inner)):
    
    norm_inner[i,:] = y_bins - freezing_inner[i]



norm_outer = np.ones((ku_outer.shape))*np.nan

for i in range(len(freezing_outer)):
    
    norm_outer[i,:] = y_bins - freezing_outer[i]
    


norm_bands = np.ones((ku_bands.shape))*np.nan

for i in range(len(freezing_bands)):
    
    norm_bands[i,:] = y_bins - freezing_bands[i]
    
    
    
#("norm bands shape and non-nan")
#print(len(norm_bands))
#print(len(norm_bands[~np.isnan(norm_bands)]))
#%%       
        
#Okay, now we have normalized altitude w.r.t 0 degC isotherm.         

'''

fig,ax = plt.subplots(figsize=(14,14)) 
xtick_label_size = 20
ytick_label_size = 20
tick_label_size = 26
title_size = 28


xlabels = np.arange(10,65,5)
#ylabels = np.arange(0,13.5,0.5)
ylabels = np.arange(-6,6,0.5)

plt.xticks(xlabels)
plt.yticks(ylabels)


plt.xlim(10,55)
plt.ylim(-6,6)


plt.ylabel('Normalized Altitude (km)', size = 26)
plt.xlabel('dBZ', size = 26)

ax.set_xticklabels(['10','15','20', '25', '30', '35', '40', '45', '50','55','60'], size = xtick_label_size)
plt.yticks(size =ytick_label_size )        
plt.title('04/14 Eyewall KuPR Profiles', size = title_size)

        
for i in range(len(freezing_eyewall)):

    plt.plot(ku_eyewall[i,:], norm_eyewall[i,:])  
    #plt.show()


fig,ax = plt.subplots(figsize=(14,14)) 
xlabels = np.arange(10,65,5)
#ylabels = np.arange(0,13.5,0.5)
ylabels = np.arange(-6,6,0.5)

plt.xticks(xlabels)
plt.yticks(ylabels)


plt.xlim(10,55)
plt.ylim(-6,6)


plt.ylabel('Normalized Altitude (km)', size = 26)
plt.xlabel('dBZ', size = 26)

ax.set_xticklabels(['10','15','20', '25', '30', '35', '40', '45', '50','55','60'], size = xtick_label_size)
plt.yticks(size =ytick_label_size )        
plt.title('04/14 Inner Core KuPR Profiles', size = title_size)

        
for i in range(len(freezing_inner)):

    plt.plot(ku_inner[i,:], norm_inner[i,:])  
    #plt.show()




fig,ax = plt.subplots(figsize=(14,14)) 
xlabels = np.arange(10,65,5)
#ylabels = np.arange(0,13.5,0.5)
ylabels = np.arange(-6,6,0.5)

plt.xticks(xlabels)
plt.yticks(ylabels)


plt.xlim(10,55)
plt.ylim(-6,6)

)
plt.ylabel('Normalized Altitude (km)', size = 26)
plt.xlabel('dBZ', size = 26)

ax.set_xticklabels(['10','15','20', '25', '30', '35', '40', '45', '50','55','60'], size = xtick_label_size)
plt.yticks(size =ytick_label_size )        
plt.title('04/14 Outer Core KuPR Profiles', size = title_size)

        
for i in range(len(freezing_outer)):

    plt.plot(ku_outer[i,:], norm_outer[i,:])  
    #plt.show()     
        





fig,ax = plt.subplots(figsize=(14,14)) 
xlabels = np.arange(10,65,5)
#ylabels = np.arange(0,13.5,0.5)
ylabels = np.arange(-6,6,0.5)

plt.xticks(xlabels)
plt.yticks(ylabels)


plt.xlim(10,55)
plt.ylim(-6,6)


plt.ylabel('Normalized Altitude (km)', size = 26)
plt.xlabel('dBZ', size = 26)

ax.set_xticklabels(['10','15','20', '25', '30', '35', '40', '45', '50','55','60'], size = xtick_label_size)
plt.yticks(size =ytick_label_size )        
plt.title('04/14 Outer Bands KuPR Profiles', size = title_size)

        
for i in range(len(freezing_bands)):

    plt.plot(ku_bands[i,:], norm_bands[i,:])  
    #plt.show()

'''
#%%

#Now let's only extract indices within the warm cloud layer; Normalized altitude between -3 and -1 km


warm_eyewall = ku_eyewall.copy()
warm_eyewall[(norm_eyewall>-1)|(norm_eyewall<-3)] = -9999
warm_eyewall = np.ma.masked_where(warm_eyewall<0,warm_eyewall)



warm_inner = ku_inner.copy()
warm_inner[(norm_inner>-1)|(norm_inner<-3)] = -9999
warm_inner = np.ma.masked_where(warm_inner<0,warm_inner)

warm_outer = ku_outer.copy()
warm_outer[(norm_outer>-1)|(norm_outer<-3)] = -9999
warm_outer = np.ma.masked_where(warm_outer<0,warm_outer)


warm_bands = ku_bands.copy()
warm_bands[(norm_bands>-1)|(norm_bands<-3)] = -9999
warm_bands = np.ma.masked_where(warm_bands<0,warm_bands)


height_warm_eyewall = norm_eyewall.copy()
height_warm_eyewall[(norm_eyewall>-1)|(norm_eyewall<-3)] = -9999
height_warm_eyewall = np.ma.masked_where(height_warm_eyewall<-10,height_warm_eyewall)

height_warm_inner = norm_inner.copy()
height_warm_inner[(norm_inner>-1)|(norm_inner<-3)] = -9999
height_warm_inner = np.ma.masked_where(height_warm_inner<-10,height_warm_inner)

height_warm_outer = norm_outer.copy()
height_warm_outer[(norm_outer>-1)|(norm_outer<-3)] = -9999
height_warm_outer = np.ma.masked_where(height_warm_outer<-10,height_warm_outer)

height_warm_bands = norm_bands.copy()
height_warm_bands[(norm_bands>-1)|(norm_bands<-3)] = -9999
height_warm_bands = np.ma.masked_where(height_warm_bands<-10,height_warm_bands)
#print("this should be greater than 10, otherwise the whole array will be masked")
#print(np.nanmax(height_warm_bands))

#Now that we have all points in the liquid layer, let's compute the slope of KuPR using linear regression

#Test for a single footprint
'''
test = ku_outer[731,:] 
test_height = height_warm_outer[731,:]
mask = ~np.isnan(test) & ~np.isnan(test_height)

slope = stats.linregress(test[mask],test_height[mask])[0]
print(slope)
'''

#1D manual linear regression fn  

def plot_linear_regression(x,y,new_x):
   
    x = np.array(x)
    y = np.array(y)

    Y = y.copy()
    X = np.ones((Y.shape[0],2))
    X[:,1] = x.copy()
    C = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
   
    Xpred = np.array((1,-3)).reshape(1,2)
    y_pred = (C[1] * new_x) + C[0]
   
    return y_pred

#Scipy? We should compare to the revised manually written fn (thank you Tomer for the above)

slope_warm_eyewall = []


for i in range(warm_eyewall.shape[0]):
    
    
    slope_eyewall = stats.mstats.linregress(warm_eyewall[i,:],height_warm_eyewall[i,:])[0]
    slope_warm_eyewall.append(slope_eyewall)


slope_warm_eyewall = np.asarray(slope_warm_eyewall)
slope_warm_eyewall[slope_warm_eyewall<-100] = np.nan

file_eyewall = np.save(str(file_index) + 'Eyewall__All_Cases.npy', slope_warm_eyewall, allow_pickle = True)
#print(slope_warm_eyewall)
#print(slope_warm_eyewall.shape)
#print("---------------")

slope_warm_inner = []


for i in range(warm_inner.shape[0]):
    
    
    slope_inner = stats.mstats.linregress(warm_inner[i,:],height_warm_inner[i,:])[0]
    slope_warm_inner.append(slope_inner)


slope_warm_inner = np.asarray(slope_warm_inner)
slope_warm_inner[slope_warm_inner<-100] = np.nan

file_inner_core = np.save(str(file_index) + 'Inner_Core_All_Cases.npy', slope_warm_inner, allow_pickle = True)


slope_warm_outer = []


for i in range(warm_outer.shape[0]):
    
    
    slope_outer = stats.mstats.linregress(warm_outer[i,:],height_warm_outer[i,:])[0]
    slope_warm_outer.append(slope_outer)


slope_warm_outer = np.asarray(slope_warm_outer)
slope_warm_outer[slope_warm_outer<-100] = np.nan


file_outer_core = np.save(str(file_index) + 'Outer_Core_All_Cases.npy', slope_warm_outer, allow_pickle = True)


slope_warm_bands = []

#print(warm_bands.shape[0])
for i in range(warm_bands.shape[0]):
    
    slope_warm_all = []
    slope_bands = stats.mstats.linregress(warm_bands[i,:],height_warm_bands[i,:])[0]
    #print(np.nanmax(warm_bands[i,:]))
    #print(np.nanmax(height_warm_bands[i,:]))
    #print(slope_bands)
    #print('----------------')
    slope_warm_bands.append(slope_bands)


slope_warm_bands = np.asarray(slope_warm_bands)
#print("this should be greater than -100 otherwise all will be nans")
#print(np.nanmax(slope_warm_bands))
slope_warm_bands[slope_warm_bands<-100] = np.nan

file_large = np.save(str(file_index) + '_All_Cases.npy', slope_warm_bands, allow_pickle = True)

#print(len(slope_warm_bands))

#%%

fontsize = 24

fig,ax = plt.subplots(figsize=(14,14)) 
sns.distplot(slope_warm_eyewall,hist = False, kde = True, bins = 20, color = 'darkblue',  hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.title('KuPR Slope in Liquid Phase of Eyewall', size = fontsize)
plt.xlabel('Slope (dBZ/km)',size = fontsize)
#ax.set_xticklabels(np.arange(-0.3,0.3,step = 0.1),size = fontsize)
plt.savefig(str(file_index) + 'Eyewall.png')
plt.show()

fig,ax = plt.subplots(figsize=(14,14)) 
sns.distplot(slope_warm_outer,hist = False, kde = True, bins = 20, color = 'darkblue',  hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.title('KuPR Slope in Liquid Phase of Outer Annulus', size = fontsize)
plt.xlabel('Slope (dBZ/km)',size = fontsize)
#ax.set_xticklabels(np.arange(-0.3,0.3,step = 0.1),size = fontsize)
plt.savefig(str(file_index) + 'Outer.png')
plt.show()


fig,ax = plt.subplots(figsize=(14,14)) 
sns.distplot(slope_warm_inner,hist = False, kde = True, bins = 20, color = 'red',  hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.title('KuPR Slope in Liquid Phase of Inner Annulus', size = fontsize)
plt.xlabel('Slope (dBZ/km)',size = fontsize)
#ax.set_xticklabels(np.arange(-0.3,0.3,step = 0.1),size = fontsize)
plt.savefig(str(file_index) + 'Inner.png')
plt.show()


fig,ax = plt.subplots(figsize=(14,14)) 
sns.distplot(slope_warm_bands,hist = False, kde = True, bins = 20, color = 'red',  hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.title('KuPR Slope in Liquid Phase of Outer Bands', size = fontsize)
plt.xlabel('Slope (dBZ/km)',size = fontsize)
#ax.set_xticklabels(np.arange(-0.3,0.3,step = 0.1),size = fontsize)
plt.savefig(str(file_index) + 'Rainbands.png')
plt.show()



#%%

#Now write an algorithm to plot on a grid footprints of positive versus negative slopes in the liquid phase:
  
'''
from matplotlib.colors import ListedColormap

colormap = ['dodgerblue','red']    

inner_colors = []

for i in range(len(slope_warm_inner)):
    
    if slope_warm_inner[i]>=0:
        inner_colors.append(colormap[0])
        
    else:
        inner_colors.append(colormap[1])


#Now we need to index lat-lon based on these points



lon_inner = lon.copy()
lat_inner = lat.copy()


lon_inner_slope = lon_inner[inner_core]
lat_inner_slope = lat_inner[inner_core]



label_size = 24
title_size = 28
tick_label_size = 20

#Setup plotting 
cmin = -2; cmax = 2; cint = 1; clevs = np.round(np.arange(cmin,cmax,cint),2)
nlevs = len(clevs) - 1; cmap = plt.get_cmap(name=colormap,lut=nlevs)

colour_norm_object = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=False)
scalar_mappable_object = plt.cm.ScalarMappable(cmap=cmap, norm=colour_norm_object)
scalar_mappable_object.set_array(z_color)

figure_object, axes_object = plt.subplots(1, 1, figsize=(10, 10))

c = plt.scatter(lon_inner_slope, lat_inner_slope, c = inner_colors, cmap = cmap, edgecolors = 'none')

'''






#%%
'''
#Lets plot this file on a grid

from mpl_toolkits.basemap import Basemap

ind1 = np.where((lon[:,0]>=95)) #Where are the DPR lons >= -100
ind2 = np.where((lon[:,0])<=99) #Where are lons <= -85
ind3 = np.intersect1d(ind1,ind2) #Both conditions need to be satisfied here



#Determine median value of ind3 (intersection between longitudes)
cross_track_index = int(len(ind3)/2)+ind3[0]



#Let's plot a map with the center point location for our cross-sections

#Mask near surface reflectivity values

z = np.ma.masked_where(z<=12, z)



cmin = 12.; cmax = 70.; cint = 2.5; clevs = np.round(np.arange(cmin,cmax,cint),2)
nlevs = len(clevs) - 1; cmap = plt.get_cmap(name='turbo',lut=nlevs)
    
plt.figure(figsize=(10,10))
  
#xlim = np.array([-110,-75]); ylim = np.array([15,40])
xlim = np.array([95,100]); ylim = np.array([-16,-8])
#xlim = np.array([35,55]); ylim = np.array([-30,-10])
 

  
m = Basemap(projection='cyl',lon_0=np.mean(xlim),lat_0=np.mean(ylim),llcrnrlat=ylim[0],urcrnrlat=ylim[1],llcrnrlon=xlim[0],urcrnrlon=xlim[1],resolution='i')
m.drawstates(), m.drawcountries(), m.drawcoastlines()
cs = m.contourf(lon,lat,z,clevs,cmap='turbo',extend='both')
cs2 = m.plot(lon[:,0]+0.03,lat[:,0]-0.03,'--k',zorder=4)
cs3 = m.plot(lon[:,-1]-0.03,lat[:,-1]+0.03,'--k',zorder=4)
#m.plot(lon[ind3,ray],lat[ind3,ray],'-w',zorder=4,lw=0.25,alpha=0.75)
#m.plot(lons[0:160],lats[0:160], 'k--', linewidth = 4)


##Change this to modify star size (cross-section starting point)
#m.plot(lon[ind3[0],ray],lat[ind3[0],ray],'*w',markersize = 30, zorder=4,lw=0.25,alpha=0.75,markerfacecolor='k',markeredgewidth=0.25)
#print(lon[ind3[0],ray])
#print(lat[ind3[0],ray])
x,y = case_lon,case_lat
m.plot(x,y,'r*',markersize=26, color = 'k')

m.plot(lon[cross_track_index,:],lat[cross_track_index,:],'-w',zorder=4,lw=0.25,alpha=0.75)
m.plot(lon[cross_track_index,0],lat[cross_track_index,0],'*w', zorder=4,lw=0.25,alpha=0.75,markerfacecolor='k',markeredgewidth=0.25)
m.drawcounties()
parallels = np.arange(-60,60,step = 2)
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
   
#plt.title('GPM Overpass 02/22 0042 UTC KuPR', size = 20)    













#%%


#Plot on a grid

from mpl_toolkits.basemap import Basemap



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




#Establish a lat-lon grid from -180 to 180, -90 to 90 in one degree intervals


lon_grid = np.arange(-180.,180., step = 1)

lat_grid = np.arange(-90.,90., step = 1)

#Now mesh to a grid

lon2,lat2 = np.meshgrid(lon_grid,lat_grid)

#Round lat and lon to nearest integers

lon_rounded = np.round(lon)
lat_rounded = np.round(lat)

plt.figure(figsize=(14,14))

m = Basemap(llcrnrlon =-180 , llcrnrlat = -60, urcrnrlon = 180, urcrnrlat = 60)
m.drawcoastlines()
m.drawstates()
m.drawcountries()

m.plot(gpm_lon,gpm_lat, 'ro', markersize = 2)

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

plt.title('GPM DPR Overpasses at Tropical Cyclone Locations (2014-2020) (Annuli)', size = font_size)


plt.show()


'''


        
    



