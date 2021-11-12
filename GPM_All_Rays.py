#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:52:06 2021

@author: noahbrauer
"""

import h5py 
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
import glob
from mpl_toolkits.basemap import Basemap






files = glob.glob('*.HDF5')


gpm_lat = np.array([14.5,25.2,-17.69,14,17])
gpm_lon = np.array([176,-112,117.6,131,123])

radius = 3
xlim = np.array([gpm_lon-radius,gpm_lon+radius])  ; ylim = np.array([gpm_lat-radius,gpm_lat+radius]) 
   
xlim_west = xlim[0]
xlim_east = xlim[1]
ylim_south = ylim[0]
ylim_north = ylim[1]


updated_list = []


for file in range(len(files)):
    data = h5py.File(files[file]) #Open files
    print(file)

    
    #Read in files, extract lat, lons, KuPR

    lat = data['NS']['Latitude'][:]
    lon = data['NS']['Longitude'][:]
    ku = data['NS']['SLV']['zFactorCorrected'][:]
    

    #7933 is the minimum swath distance value in all files. Can switch to np.min(ku.shape[0])
    
    
    if ku.shape[0] != 7933:
        
        diff = ku.shape[0] - 7933  #Compute difference between minimum shape value and actual shape value of all files
        print(diff)
        
        updated_case = ku[:-diff,:,:] #Slice off the end by the difference value
        print(updated_case.shape)
        
        updated_list.append(updated_case) 
        
    else:
        
        updated_list.append(updated_case) #If no difference, then keep KuPR the same
    


array_kupr = np.stack(updated_list)



#%%

for i in range(len(xlim_west)):    

   ind1 = np.where((lon[:,0]>=xlim_west[i])) #Where are the DPR lons >= -100
   ind2 = np.where((lon[:,0])<=xlim_east[i]) #Where are lons <= -85
   ind3 = np.intersect1d(ind1,ind2) #Both conditions need to be satisfied here
   
   x = 2.* 17 #48 degrees (from -17 to 17)
   re = 6378. #radius of the earth
   theta = -1*(x/2.) + (x/48.)*np.arange(0,49) #Split into equal degrees (from -17 to 17)
   theta2  = np.ones(theta.shape[0]+1)*np.nan #Define an empty array (NaNs) with same shape as ray dimension
   theta = theta - 0.70833333/2. #Shift degrees for plotting pruposes
   theta2[:-1] = theta #remove last array dimension in theta so python won't get angry about shape
   theta2[-1] = theta[-1] + 0.70833333
   theta = theta2*(np.pi/180.) #Convert from degrees to radians

   prh = np.ones((177,50))*np.nan #Define empty grid

   for i in range(prh.shape[0]): #Loop over each range gate
       for j in range(prh.shape[1]): #Loop over each scan
            a = np.arcsin(((re+407)/re)*np.sin(theta[j]))-theta[j] #Orbit height of 407 km

            prh[i,j] = (176-(i))*0.125*np.cos(theta[j]+a)

   h2 = prh #where prh is the (range bins,ray)-space
   h3 =np.ones((177,50))*np.nan

   for i in range(h3.shape[1]):
       h3[:,i] = h2[::-1,i] #This reverses the vertical dimension so as indices increase, height increases
        

kupr = array_kupr[:,ind3,:,:]

lons = data['NS']['Longitude'][ind3,:]
lats = data['NS']['Latitude'][ind3,:]
    















    
