#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:52:57 2021

@author: noahbrauer
"""


import matplotlib.pyplot as plt
import numpy as np


import wget
import requests

file = 'orbit_no.txt'

data = np.loadtxt(file, skiprows = 2)


orbit_2014 = data[0:49]
orbit_2015 = str(data[50:131])
orbit_2016 = data[132:194]
orbit_2017 = data[195:261]
orbit_2018 = data[262:337]
orbit_2019 = data[338:431]
orbit_2020 = data[432:]



#Now pull a file from one url:

    
URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.06/2014/070/"
    
FILENAME = "2A.GPM.DPR.V8-20180723.20140311-S010040-E023305.000177.V06A.HDF5"
    
    
#Retrieve URL 
    
result = requests.get(URL)
    
try:
    result.raise_for_status()
        
    f = open(FILENAME,'wb')
    f.write(result.content)
    f.close()
        
    print('Contents of URL written to' +FILENAME)
        
except:
    print('requests.get() returned an error code' + str(result.status_code()))
    

#Cool, it works. Now lets try with multiple files.
#%%

url_number = np.arange(1,366,step = 1)

def str3(num):
    
    if num < 10: return f"00{num}"
    
    if num < 100: return f"0{num}"
    
    return str(num)



url_list = []


for i in range(len(url_number)):
    
    digits = str3(url_number[i])
    
    url_list.append(digits)
    
        
#%%                  
import re


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
        
        
        
        