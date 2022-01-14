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
import pandas as pd
#from datetime import datetime
import datetime as dt




files = glob.glob('*.HDF5')




#Read in file, extract lat, lons, PSD

gpm_coords_file = 'output.csv'
data_csv = pd.read_csv(gpm_coords_file,usecols = ['gpm_lat','gpm_lon','storm_date', 'storm_lat','storm_lon'])




gpm_lat_series = data_csv['gpm_lat'][:]
gpm_lon_series = data_csv['gpm_lon'][:]
storm_time = data_csv['storm_date'][:]
storm_lat_series = data_csv['storm_lat'][:]
storm_lon_series = data_csv['storm_lon'][:]

#Convert series to numpy arrays

gpm_lat = pd.Series.to_numpy(gpm_lat_series)
gpm_lon = pd.Series.to_numpy(gpm_lon_series)
storm_time_array = pd.Series.to_numpy(storm_time)
storm_lat = pd.Series.to_numpy(storm_lat_series)
storm_lon = pd.Series.to_numpy(storm_lon_series)

#Now we need to read in the start and end times from GPM --> then convert to datetime objects


start_time_gpm = []
end_time_gpm = []

for i in range(len(files)):

    period_split = files[i].split(".")[4]


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


