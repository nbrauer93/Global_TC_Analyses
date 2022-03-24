#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 12:44:03 2022

@author: noahbrauer
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
import glob
from mpl_toolkits.basemap import Basemap
import pandas as pd
import datetime as dt




files = glob.glob('*.HDF5')


files_sort = np.argsort(files)



#Read in file, extract lat, lons, PSD

gpm_coords_file = 'output.csv'
data_csv = pd.read_csv(gpm_coords_file,usecols = ['gpm_lat','gpm_lon','storm_date', 'storm_lat','storm_lon', 'gpm_date'])




gpm_lat_series = data_csv['gpm_lat'][:]
gpm_lon_series = data_csv['gpm_lon'][:]
storm_time = data_csv['storm_date'][:]
storm_lat_series = data_csv['storm_lat'][:]
storm_lon_series = data_csv['storm_lon'][:]
gpm_date = data_csv['gpm_date'][:]

#Convert series to numpy arrays

gpm_lat = pd.Series.to_numpy(gpm_lat_series)
gpm_lon = pd.Series.to_numpy(gpm_lon_series)
storm_time_array = pd.Series.to_numpy(storm_time)
storm_lat = pd.Series.to_numpy(storm_lat_series)
storm_lon = pd.Series.to_numpy(storm_lon_series)
gpm_time = pd.to_datetime(gpm_date)
julian_time = gpm_time.apply(pd.Timestamp.toordinal) + 1721424.5
#Now we need to read in the start and end times from GPM --> then convert to datetime objects


#%%
'''
#Remove files where latitude > 40N and 40S

retain_values = np.where((gpm_lat<40)&(gpm_lat>-40))[0]


gpm_lat_qc = gpm_lat[retain_values]
gpm_lon_qc = gpm_lon[retain_values]



files_qc = []

for i in retain_values:
    files_qc.append(files[i])


#So now we have a QC-ed list of files based off latitude
#Now, let's order the files based off time: 
    
files_qc_order = np.argsort(files_qc)    


#Then index the list to reflect this time-ordered list:
    
    
files_ordered = [files_qc[i] for i in files_qc_order]

  


#Read in the actual DPR files:

for file in range(len(files_ordered)):
    
    
    
    
    data = h5py.File(files_ordered[file])
    #print(file)
    print(files_ordered[file])
    #print(gpm_lat_sort[file])
    #print(gpm_lon_sort[file])

    z_corrected = data['NS']['SLV']['zFactorCorrectedNearSurface'][:]
  





    lat = data['NS']['Latitude'][:]
    lon = data['NS']['Longitude'][:]
#Lat and lon corresponding to the DPR overpass locations

    domain_range = 6


    z_corrected[z_corrected<20] = np.nan

    cmin = 12.; cmax = 60.; cint = 2.5; clevs = np.round(np.arange(cmin,cmax,cint),2)
    nlevs = len(clevs) - 1; cmap = plt.get_cmap(name='turbo',lut=nlevs)

    plt.figure(figsize=(20,20))
    
    m = Basemap(projection='cyl',lon_0=gpm_lon_qc[file],lat_0=gpm_lat_qc[file],llcrnrlat=gpm_lat_qc[file]-domain_range,urcrnrlat=gpm_lat_qc[file]+domain_range,llcrnrlon=gpm_lon_qc[file]-domain_range,urcrnrlon=gpm_lon_qc[file]+domain_range,resolution='l')
    m.drawcoastlines(); m.drawstates(), m.drawcountries()


    cs = m.contourf(lon,lat,z_corrected,clevs,cmap='turbo',extend='both')
    cs2 = m.plot(lon[:,0]+0.03,lat[:,0]-0.03,'--k',zorder=1)
    cs3 = m.plot(lon[:,-1]-0.03,lat[:,-1]+0.03,'--k',zorder=1)
    
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



    plt.title('GPM Overpass KuPR', size = 20)
    plt.show()
    
'''
#%%

#####Comment this back in to account for latitude threshold ########


'''
retain_values = np.where((gpm_lat<40)&(gpm_lat>-40))[0]


gpm_lat_qc = gpm_lat[retain_values]
gpm_lon_qc = gpm_lon[retain_values]



files_qc = []

for i in retain_values:
    files_qc.append(files[i])


#So now we have a QC-ed list of files based off latitude
#Now, let's order the files based off time: 
'''    
files_qc_order = np.argsort(files)    


#Then index the list to reflect this time-ordered list:
    
    
files_ordered = [files[i] for i in files_qc_order]



#Now let's parse the date out of the file name --> Add to title


date_object = []


for i in range(len(files_ordered)):
    
     period_split = files_ordered[i].split(".")[4]
     print(period_split)
     date_object.append(period_split)
    
    
    
    
  
#%%

#Read in the actual DPR files:

for file in range(len(files_ordered)):
    
    
    
    
    data = h5py.File(files_ordered[file])
    #print(file)
    print(files_ordered[file])
    #print(gpm_lat_sort[file])
    #print(gpm_lon_sort[file])

    z_corrected = data['NS']['SLV']['zFactorCorrectedNearSurface'][:]
  





    lat = data['NS']['Latitude'][:]
    lon = data['NS']['Longitude'][:]
#Lat and lon corresponding to the DPR overpass locations

    domain_range = 6


    z_corrected[z_corrected<20] = np.nan

    cmin = 12.; cmax = 70.; cint = 2.5; clevs = np.round(np.arange(cmin,cmax,cint),2)
    nlevs = len(clevs) - 1; cmap = plt.get_cmap(name='turbo',lut=nlevs)

    plt.figure(figsize=(20,20))
    
    m = Basemap(projection='cyl',lon_0=gpm_lon[file],lat_0=gpm_lat[file],llcrnrlat=gpm_lat[file]-domain_range,urcrnrlat=gpm_lat[file]+domain_range,llcrnrlon=gpm_lon[file]-domain_range,urcrnrlon=gpm_lon[file]+domain_range,resolution='l')
    m.drawcoastlines(); m.drawstates(), m.drawcountries()


    cs = m.contourf(lon,lat,z_corrected,clevs,cmap='turbo',extend='both')
    cs2 = m.plot(lon[:,0]+0.03,lat[:,0]-0.03,'--k',zorder=1)
    cs3 = m.plot(lon[:,-1]-0.03,lat[:,-1]+0.03,'--k',zorder=1)
    
    parallels = np.arange(-90,90,step = 2)
    m.drawparallels(parallels, labels = [True, False, False, False])

    meridians = np.arange(0,360, step = 2)
    m.drawmeridians(meridians, labels = [False, False, False, True])

#Plot storm location (storm_lat, storm_lon)

    x2star,y2star = m(storm_lon[file],storm_lat[file])
    m.plot(x2star,y2star,'r*',markersize=26, color = 'k')


    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 24)
    cbar.set_label(label = '[dBZ]',size = 24)
    plt.clim(12,70)



    plt.title('GPM Overpass KuPR ' + date_object[file], size = 30)
    plt.show()













#%%

'''
start_time_gpm = []
end_time_gpm = []


for i in range(len(files)):

    period_split = files[i].split(".")[4]
    print(period_split)


    #Now in the format of YYMMDD-SHHMMSS-EHHMMSS
    #Need to remove the S and E from the strings

    remove_characters = "SE"
    time = period_split

    for letter in remove_characters:

        time = time.replace(letter,"")
        #time has format YYMMDD-HHMMSS-HHMMSS

    #files_time.append(time)
    #Split into YYMMDD, Start HHMMSS, End HHMMSS
    date = time.split("-",3)[0]
    start = time.split("-",3)[1]
    end = time.split("-",3)[2]

    #Now define start time including date: YYMMDD HHMMSS; Same with end time

    start_time = str(date) + ' '  + str(start)
    end_time = str(date) + ' ' + str(end)
    #Now we need to format in terms of datetime objects

    datetime_start = dt.datetime.strptime(start_time, "%Y%m%d %H%M%S")
    datetime_end = dt.datetime.strptime(end_time, "%Y%m%d %H%M%S")
   

    #Convert to Julian Time:

    julian_start = datetime_start.toordinal() + 1721424.5
    julian_end = datetime_end.toordinal() + 1721424.5
  

    start_time_gpm.append(julian_start)
    end_time_gpm.append(julian_end)
    
#Change times to arrays from lists:



    

start_time_gpm = np.array(start_time_gpm, dtype = float)
end_time_gpm = np.array(end_time_gpm, dtype = float)





#Now put start and end times to a 356x2 array:

gpm_time_interval = np.vstack((start_time_gpm, end_time_gpm)).T

#Cool; Now we have the GPM DPR times in a reasonable format. Let's deal with the HURDAT2 data now


#Now convert HURDAT2 times to datetime objects
#Has format YYYY-mm-DD HH:MM:SS

track_time = []

for i in range(len(storm_time_array)):

    ibtrack_time = dt.datetime.fromisoformat(storm_time_array[i])
    storm_julian = ibtrack_time.toordinal() + 1721424.5
    track_time.append(storm_julian)

storm_time = np.array(track_time)


#Time index. Find the cases where GPM overpass time is between 
#Extract times from each DPR file --> Convert to datetime objects



files_date = []
files_datetime = []
files_julian = []


for i in range(len(files)):
    
    splitted = files[i].split("-", 4)[1]
    print(splitted)
    #Now split at period to retain onlt the date

    splitted_junk = splitted.split(".")[1]
    
    files_date.append(splitted_junk)
    
    
    
    datetime_object = dt.datetime.strptime(splitted_junk,"%Y%m%d")
    files_datetime.append(datetime_object)
    
    julian_file = datetime_object.toordinal() + 1721424.5
    files_julian.append(julian_file)
    







#%%
#Now that we have the GPM DPR start and end times along with the storm track time
#Find where GPM lat-lon is centered within a range of storm_lat and storm_lon (3 degrees will be the default)

degree_range = 3

lat_domain_index = np.where((gpm_lat>storm_lat-degree_range)&(gpm_lat<storm_lat+degree_range))[0]


#Longitude ranges from -180 to 180


lon_domain_index = np.where((gpm_lon>storm_lon-degree_range)&(gpm_lon<storm_lon+degree_range))[0]

#Now set the domain:

lon_domain = gpm_lon[lon_domain_index]
lat_domain = gpm_lat[lat_domain_index]

print(lat_domain)
print(lon_domain)
'''
    