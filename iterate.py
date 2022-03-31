#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:19:58 2022

@author: noahbrauer
"""



import os
import glob
import numpy as np


files = glob.glob('*.HDF5')
#order files here
order = np.argsort(files)
files = [files[i] for i in order[:8]]


for idx,file in enumerate(files):
    os.system(f"python Read_Shear.py {file} {idx}")