#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

radius = 3
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
    
#%%



 ###Define cross-section length

for i in range(len(xlim_west)):    

   ind1 = np.where((lon[:,0]>=xlim_west[i])) #Where are the DPR lons >= -100
   ind2 = np.where((lon[:,0])<=xlim_east[i]) #Where are lons <= -85
   ind3 = np.intersect1d(ind1,ind2) #Both conditions need to be satisfied here
    
    ###Setup to 2D grid for plotting

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
        
   ku = data['NS']['SLV']['zFactorCorrected'][ind3,:,:] #Read in ku-band reflectivity; nscan x nray (554,49,176)
    
   ray = 27

   ku = ku[:,ray,:]
   #Take lats and lons along same ray
   lons = data['NS']['Longitude'][ind3,ray]
   lats = data['NS']['Latitude'][ind3,ray]
    
   #Choose a starting point, then calculate distance
   lat0 = lats[0]
   lon0 = lons[0]
    
   p = Proj(proj='laea', zone=10, ellps='WGS84',lat_0=lat0,lon_0=lon0) #Define a projection and set starting lat an lon to same point as above

#Define a 2D array for plotting purposes

   lat_3d = np.ones(ku.shape)*np.nan
   lon_3d = np.ones(ku.shape)*np.nan
    
   for i in range(ku.shape[0]):
       lat_3d[i,:] = lats[i]
       lon_3d[i,:] = lons[i]
        
   x,y = p(lon_3d,lat_3d) #Now convert degrees to distance (in km)
   R_gpm = np.sqrt(x**2 + y**2)*np.sign(x) #Keeps sign of number; converts to radial distance

    #Reverse range gate order for all parameters

   ku = ku[:,::-1]
    
   ku = np.ma.masked_where(ku<=12, ku) #Mask all the bad points in ku data
   y = np.ones([ku.shape[0], ku.shape[1]]) #Define an empty array

    #Define the appropriate range bins
   h4 = h3[:,ray] #This number MUST match the same ray being used
   for i in range(y.shape[1]):
       y[:,i] = h4[i]
        
        
    #Now we plot
    
   plt.figure(figsize=(10,10))

   vmax = 60
   vmin =12
   label_size = 20
    
   dist = np.array([0,1000])

   R_min = R_gpm.min()
   R_max = R_gpm.max()



   cmin =12.; cmax = 60.; cint = 2.5; clevs = np.round(np.arange(cmin,cmax,cint),2)
   nlevs = len(clevs) - 1; cmap = plt.get_cmap(name='turbo',lut=nlevs)
   pm = plt.pcolormesh(R_gpm/1000., y, ku, cmap='turbo',vmin=vmin,vmax=vmax)
   #pm2 = plt.plot(R_gpm/1000., zero_deg_isotherm, '--', color = 'k')
   plt.xlabel('Along Track Distance (km)', size = 20)
   plt.ylabel('Altitude (km)', size = 20)
   plt.title(r'GPM Overpass 8/27 0301 UTC KuPR ', size = 20)
   #plt.xlim(300,450)
   plt.xlim(dist[0], dist[1])
   plt.ylim(0,15)
   cbar = plt.colorbar()
   cbar.ax.tick_params(labelsize = label_size)
   cbar.set_label(label = '[dBZ]',size = label_size)
   plt.clim(12,60)
   plt.xticks(size = label_size)
   plt.yticks(size = label_size)


   plt.show()






