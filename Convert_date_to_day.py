#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:36:55 2021

@author: noahbrauer
"""

import matplotlib.pyplot as plt
import numpy as np
#from datetime import datetime, date
import datetime
import pandas as pd




file = 'output.csv'

data = pd.read_csv(file)

gpm_date = data['gpm_date'][:]



gpm_np = pd.Series.to_numpy(gpm_date)



def date_to_nth_day(date, format="%Y-%m-%d  %H:%M:%S"):
    date = pd.to_datetime(date, format=format)
    new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
    return (date - new_year_day).days + 1


day = []

for i in range(len(gpm_date)):
    
    day_of_year = date_to_nth_day(gpm_date[i])
    
    day.append(day_of_year)
    
    
#Add zeros in front over values less than 100


day = np.asarray(day, dtype = 'str')

date_format = []


for i in range(len(day)):
    
    with_zeros = day[i].zfill(3)
    
    date_format.append(with_zeros)
    
    
#Remove the first two entries as these are bad and NaNs

date_format_parsed = date_format[2:]    



#%%
#From datetime objects, remove HH, MM, and hyphens

date_only = [i[0:10] for i in gpm_np]


yymmdd = []

for i in date_only:
    
    yymmdd.append(i.replace('-', ''))



#%%

    
  
#Okay now the time is properly formatted. Let's import the orbit numbers


orbit_file = 'orbit_no.txt'

orbit_no = np.loadtxt(orbit_file, skiprows = 2, dtype = 'str')   



orbit_no_format = []


for i in range(len(orbit_no)):
    
    with_zeros = orbit_no[i].zfill(5)
    orbit_no_format.append(with_zeros)


#Now we have orbit numbers and dates in the proper format. Use wget to loop through each array and pull each file from the appropriate URL. 
#Let's split up the dates and orbit numbers by year

orbit_2014, date_2014 = orbit_no_format[0:49],date_format_parsed[0:49]
orbit_2015,date_2015 = orbit_no_format[50:131],date_format_parsed[50:131]
orbit_2016,date_2016 = orbit_no_format[132:194],date_format_parsed[132:194]
orbit_2017,date_2017 = orbit_no_format[195:261],date_format_parsed[195:261]
orbit_2018,date_2018 = orbit_no_format[262:337],date_format_parsed[262:337]
orbit_2019,date_2019 = orbit_no_format[338:431],date_format_parsed[338:431]
orbit_2020,date_2020 = orbit_no_format[432:502],date_format_parsed[432:502]

 
#import re
#import wget

#This is just for 2014

year = ['2014','2015','2016','2017','2018','2019','2020']



year_url = []

for i in range(len(year)):
    
    URL = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.06/' + year[i] + '/'
    year_url.append(URL)
    
    
#%%






io = drpy.io.netrunner(servername='Research',username='nbrauer@ou.edu',start_time=datetime.datetime(2015,12,3,15,9),Xradar=False, savedir = '/Users/noahbrauer/Desktop/Global_TCs/DPR_Files/')


#%%

import datetime
import drpy 

#datetime_dpr = pd.to_datetime(gpm_date[11:])
datetime_dpr = pd.to_datetime(gpm_date[11:])

for date in datetime_dpr:
    
    print(date)
    
    io = drpy.io.netrunner(servername='Research',username='nbrauer@ou.edu',start_time=date,Xradar=False, savedir = '/Users/noahbrauer/Desktop/DPR_Files/')

    
#Xradar argument; False: KaPR inner swath  #FS in full KaPR swath
#%%

'''
for j in range(len(date_2014)):
    
    filename = '2A.GPM.DPR.V8-.+.' + orbit_2014[j] + 'V06A.HDF5'
    
    path = year_url[0] + date_2014[j] + filename 
    wget.download(path, out = '/Users/noahbrauer/Desktop/Global_TCs/' + year[0] + '/')
    




#Now loop through all directories. Start with 2015 orbit numbers

for i in range(0,365):
    for j in range(len(orbit_2015)):
        
        url = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.06/2015/' + url_list[i]+'/'
        
        filename = '2A.GPM.DPR.V8-.+.' + orbit_2015[j] + 'V06A.HDF5'
        
        result = requests.get(url)
    
        try:
            result.raise_for_status()
        
            f = open(filename,'wb')
            f.write(result.content)
            f.close()
        
            print('Contents of URL written to' +filename)
        
        except:
            print('requests.get() returned an error code' + str(result.status_code()))
'''       
        
        
 #%%
 '''
 
 #date_2014 and orbit_2014 have the same length  (49)
 #year_url has length 7

for i in range(len(orbit_2014)):
    for j in range(len(yymmdd)):
    
        url =  year_url[0] 
        
        filename = '2A.GPM.DPR.V8-20180723.' + yymmdd[j] + '-' + 
        
 '''      
    
    



       
    
    
    












    
    

    