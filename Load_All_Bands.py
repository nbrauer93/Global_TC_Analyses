#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:55:24 2022

@author: noahbrauer
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns

files = glob.glob('*.npy')


all_bands = []

for i in range(len(files)):
    
    data = np.load(files[i], allow_pickle = False)
    print(data)
    all_bands.append(data)
    

all_bands_array = np.asarray(all_bands)

all_rainbands = np.hstack(all_bands_array)

    
    
fontsize = 24

fig,ax = plt.subplots(figsize=(14,14)) 
sns.distplot(all_rainbands,hist = False, kde = True, bins = 20, color = 'darkblue',  hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.xlim(-3,3)
plt.title('KuPR Slope in Liquid Phase of Outer Bands (149 cases; n = 45681)', size = fontsize)
plt.xlabel('Slope (dBZ/km)',size = fontsize)
plt.ylabel('Density', size = fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize = fontsize)
plt.show()    



#%%

files_inner = glob.glob('*Inner_Core*')
files_outer = glob.glob('*Outer_Core*')
files_eyewall = glob.glob('*Eyewall*.npy')


all_outer = []

for i in range(len(files_outer)):
    
    data = np.load(files_outer[i], allow_pickle = False)
    print(data)
    all_outer.append(data)
    

all_outer_array = np.asarray(all_outer)

all_outer_core = np.hstack(all_outer_array)

    
    
fontsize = 24

fig,ax = plt.subplots(figsize=(14,14)) 
sns.distplot(all_outer_core,hist = False, kde = True, bins = 20, color = 'darkblue',  hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.xlim(-7,7)
plt.title('KuPR Slope in Liquid Phase of Outer Core (149 cases; n = 20505)', size = fontsize)
plt.xlabel('Slope (dBZ/km)',size = fontsize)
plt.ylabel('Density', size = fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize = fontsize)
plt.show()    



#%%




all_inner = []

for i in range(len(files_inner)):
    
    data = np.load(files_inner[i], allow_pickle = False)
    print(data)
    all_inner.append(data)
    

all_inner_array = np.asarray(all_inner)

all_inner_core = np.hstack(all_inner_array)

    
    
fontsize = 24

fig,ax = plt.subplots(figsize=(14,14)) 
sns.distplot(all_inner_core,hist = False, kde = True, bins = 20, color = 'red',  hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.xlim(-7,7)
plt.title('KuPR Slope in Liquid Phase of Inner Core (149 cases; n = 9471)', size = fontsize)
plt.xlabel('Slope (dBZ/km)',size = fontsize)
plt.ylabel('Density', size = fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize = fontsize)
plt.show()    


#%%


all_eyewall = []

for i in range(len(files_eyewall)):
    
    data = np.load(files_eyewall[i], allow_pickle = False)
    print(data)
    all_eyewall.append(data)
    

all_eyewall_array = np.asarray(all_eyewall)

all_eyewall_profiles = np.hstack(all_eyewall_array)

    
    
fontsize = 24

fig,ax = plt.subplots(figsize=(14,14)) 
sns.distplot(all_eyewall_profiles,hist = False, kde = True, bins = 20, color = 'red',  hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.xlim(-7,7)
plt.title('KuPR Slope in Liquid Phase of Eyewall (149 cases; n = 736)', size = fontsize)
plt.xlabel('Slope (dBZ/km)',size = fontsize)
plt.ylabel('Density', size = fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize = fontsize)
plt.show()    



