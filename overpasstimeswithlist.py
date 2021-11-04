import numpy as np
import datetime
import re
import sys, os
from pyorbital.orbital import Orbital
from math import sin, cos, sqrt, atan2, radians


#Convert lat-lon distance between two coordinates to km

def distance_km(lat1, lon1, lat2, lon2):

  R = 6373.0
  lat1= radians(lat1)
  lon1= radians(lon1)
  lat2= radians(lat2)
  lon2= radians(lon2)
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  km= R * c

  return(km)

#Function to get two line elements (TLE)? 

def get_TLE(tle_list, storm_date, sat_name, found_tle_file):

  #print("Reading from ", tle_list)
  lines = []
  with open(tle_list, 'r') as tle_file: #Loop through all files in file list and open
    for line in tle_file:
      lines.append(line)
  tle_file.close()
  #print(lines)

  maxdays= 30  # look for TLE within this many days  #Specify a date range 
  dt1 = datetime.datetime.strptime(str(storm_date), "%Y-%m-%d %H:%M:%S")

  #--- Find a TLE nearby to this day

#Find a TLE 

  found= -1
  day= 0
  if ( os.path.exists(found_tle_file) ): os.remove(found_tle_file)

  while day <= 2*maxdays:
    if ( found > 0 ): break
    if ( int(day % 2) != 0 ): sign= -1
    else: sign= 1
    day_next= sign*int((day+1)/2)
    tleday= dt1 + datetime.timedelta(hours=24*day_next)
    #tle_ID = tleday.strftime("' %y%j\.'")
    tle_ID = tleday.strftime(" %y%j.")
    print("Searching " + tle_list + " for ", tle_ID)

    for i in range(0,len(lines)):
      if ( i % 2 != 0 ): continue
      line1= lines[i].rstrip('\n')
      if ( line1.find(tle_ID, 17, 24) == -1 ): continue
      found= 1
      line2= lines[i+1].rstrip('\n')
      print(line1)
      print(line2)

      f= open(found_tle_file, "w")
      f.write(sat_name)
      f.write("\n")
      f.write(line1)
      f.write("\n")
      f.write(line2)
      f.write("\n")
      f.close()
      break

    day+= 1


  if ( found < 0 ):
    print("No TLE found after ", maxdays, " days");
    return False
  else:
    return(found_tle_file)


def find_overpasstimes(rscat, storm_name, storm_lat, storm_lon, storm_date, timewindow, sat_name, sat_swath):

  dt0 = datetime.datetime.strptime(storm_date, "%Y-%m-%d %H:%M:%S")
  dt1= dt0 + datetime.timedelta(hours=-1*timewindow)
  dt2= dt0 + datetime.timedelta(hours=timewindow)

  print("%s %8.3f %8.3f %s" % (storm_name, storm_lat, storm_lon, storm_date))

  passes = []

  #-- coarse search
  n= nswath= 0
  dt= dt1

  while ( dt < dt2 ):
    dt= dt1 + datetime.timedelta(seconds=n)
    sdate = dt.strftime("%Y-%m-%d %H:%M:%S")
    xyz= rscat.get_lonlatalt(dt)

    km= distance_km(xyz[1], xyz[0], storm_lat, storm_lon)

    #if ( nswath == 0 and km <= sat_swath ):
    if ( km <= sat_swath ):
      lat_s= xyz[1]
      lon_s= xyz[0]
      if ( nswath == 0 ): sdt1_fine= dt
      nswath+= 1
    elif ( nswath > 0 and km > sat_swath ):
      sdt2_fine= dt

      #-- fine search
      sdate1 = sdt1_fine.strftime("%Y-%m-%d %H:%M:%S")
      sdate2 = sdt2_fine.strftime("%Y-%m-%d %H:%M:%S")
      print("N=",nswath,"  Coarse search between ", sdate1, sdate2)

      sdt1= sdt1_fine
      sdt2= sdt2_fine
      sdt= sdt1
      sn= 0
      km_min= 1E8
      km3= [1E8,1E8,1E8]

      while ( sdt < sdt2 ):
        sdt= sdt1 + datetime.timedelta(seconds=sn)
        sdate = sdt.strftime("%Y-%m-%d %H:%M:%S")
        xyz= rscat.get_lonlatalt(sdt)

        km= distance_km(xyz[1], xyz[0], storm_lat, storm_lon)

        #--- local minimum
        km3[0]= km3[1]
        km3[1]= km3[2]
        km3[2]= km
        km= km3[1]

        #print(sdate, xyz, km)
        if ( km < km3[0] and km < km3[2] ):
        #if ( km < km_min ):
          km_min= km
          lat_s= xyz[1]
          lon_s= xyz[0]
          dt_s= sdt
          #print(km3)
          if ( km_min < 0.50*sat_swath ):
            sdate = dt_s.strftime("%Y-%m-%d %H:%M:%S")
            print("%10s %8.3f %8.3f %s %8.3f" % (sat_name, lat_s, lon_s, sdate, km_min))
            opass = [sat_name, lat_s, lon_s, sdate, km_min]
            #opass = ['1', '2', '3', '4']
            passes.append(opass)

        sn+= 1
      nswath= 0
    n+= 60

  if ( len(passes) == 0 ):
    print("No coverage")
    return passes
  return passes



if __name__ == '__main__':

  storm_name = "SALLY"	
  storm_date = '2020-09-15 00:00:00'
  storm_lat = 30.0
  storm_lon= -99.0

  sat_name = "GPM"
  sat_swath = 800		# swath of GMI
  tle_list = "tle.GPM"		# name of TLE list file
  timewindow= 24		# Find all overpasses within +/- this many hours


  found_tle_file = "found_tle.txt"
  if ( os.path.exists(found_tle_file) ): os.remove(found_tle_file)

  tlef= get_TLE(tle_list, storm_date, sat_name, found_tle_file)

  if ( os.path.exists(found_tle_file) ):
    print("Found TLE nearby date ", storm_date)

    rscat= Orbital(sat_name, tle_file=found_tle_file)

    passes= find_overpasstimes(rscat, storm_name, storm_lat, storm_lon, storm_date, timewindow, sat_name, sat_swath)

    for idx, lpass in enumerate(passes):
      print("Overpass number ",idx, " Now get the data according to these coordinates and times ", lpass)

    #print(passes)

  if ( os.path.exists(found_tle_file) ): os.remove(found_tle_file)

#%%


#Alright comment here

'''


if __name__ == '__main__':

  storm_name = "PIERRE"	
  storm_date = '2019-09-15 00:00:00'
  storm_lat = 30.0
  storm_lon= -78.0

  sat_name = "GPM"
  sat_swath = 800		# swath of GMI
  tle_list = "tle.GPM"		# name of TLE list file
  timewindow= 24		# Find all overpasses within +/- this many hours


  found_tle_file = "found_tle.txt"
  if ( os.path.exists(found_tle_file) ): os.remove(found_tle_file)

  tlef= get_TLE(tle_list, storm_date, sat_name, found_tle_file)

  if ( os.path.exists(found_tle_file) ):
    print("Found TLE nearby date ", storm_date)

    rscat= Orbital(sat_name, tle_file=found_tle_file)

    passes= find_overpasstimes(rscat, storm_name, storm_lat, storm_lon, storm_date, timewindow, sat_name, sat_swath)

    for idx, lpass in enumerate(passes):
      print("Overpass number ",idx, " Now get the data according to these coordinates and times ", lpass)

    #print(passes)

  if ( os.path.exists(found_tle_file) ): os.remove(found_tle_file)
'''


####Now will need to import all storms from the HURDAT2 Database, including all times, lat, and lon. 
####

