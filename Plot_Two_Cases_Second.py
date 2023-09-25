#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:02:04 2023

@author: noahbrauer
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:13:37 2023

@author: noahbrauer
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.patches import Circle

#Read in the data

file = '2A.GPM.DPR.V8-20180723.20170917-S053205-E070439.020187.V06A.HDF5'

data = h5py.File(file)

lat = data['NS']['Latitude'][:]
lon = data['NS']['Longitude'][:]

z_corrected = data['NS']['SLV']['zFactorCorrectedNearSurface'][:]


storm_lat = 20.2
storm_lon = -110.3

wind_dir = 26

#Let's create a function to determine the end point of the arrow based on shear direction:

def arrow_endpoint(wind_dir):

    if 0<wind_dir<90: #NE shear

        angle_deg = 90 - wind_dir
        angle = np.deg2rad(angle_deg)
        dx = -1*np.cos(angle)
        dy = -1*np.sin(angle)

    if 90<wind_dir<180: #SE shear

        angle_deg = 180 - wind_dir
        angle = np.deg2rad(angle_deg)
        dx = -1*np.cos(angle)
        dy = np.sin(angle)

    if 180<wind_dir<270: #SW shear

        angle_deg = 270-wind_dir
        angle = np.deg2rad(angle_deg)
        dx = np.cos(angle)
        dy = np.sin(angle)

    if 270<wind_dir<360: #NW shear

        angle_deg = wind_dir - 270
        angle = np.deg2rad(angle_deg)
        dx = np.cos(angle)
        dy = -1*np.sin(angle)

    if wind_dir  == 0 or wind_dir == 360:

        dx = 0
        dy = -1

    if wind_dir == 90:

        dx = -1
        dy = 0

    if wind_dir == 180:

        dx = 0
        dy = 1

    if wind_dir == 270:

        dx = 1
        dy = 0


    return [dx,dy]


dx = arrow_endpoint(wind_dir)[0]
dy = arrow_endpoint(wind_dir)[1]

#print(dx)
#print(dy)


#Now compute and plot a perpendicular line


if dy != 0 :

    perp_slope = (-1*dx/dy)
    intercept = storm_lat+(dx/dy)*storm_lon

else:

    perp_slope = 0
    intercept = storm_lon

#print(intercept)
#print(perp_slope)


#Now set up two arbitrary arrays for lat and lon based off storm center:
#Line length refers to desired length of arrow (detertmine by radius of maximum wind?)

line_length = 1.351351

lats = np.array([storm_lat-line_length, storm_lat + line_length])
lons = np.array([storm_lon-line_length, storm_lon + line_length])


#So now to plot line perpendicular to arrow:
# x = lons
# y = (perp_slope)*lons + intercept


#Now extend the arrow down past the point
#Arrow length is simply sqrt(dx**2 + dy**2)

arrow_length = np.sqrt((dx**2)+ (dy**2))

coords_lat = np.array([storm_lat,storm_lat - dy])
coords_lon = np.array([storm_lon,storm_lon- dx])



z_corrected[z_corrected<12] = np.nan

from mpl_toolkits.basemap import Basemap

cmin = 12; cmax = 70; cint = 2; clevs = np.round(np.arange(cmin,cmax,cint),2)
nlevs = len(clevs) - 1; cmap = plt.get_cmap(name='turbo',lut=nlevs)

plt.figure(figsize=(14,14))


xlim = np.array([-114,-106]); ylim = np.array([16,24])


label_size = 24


m = Basemap(projection='cyl',lon_0=np.mean(xlim),lat_0=np.mean(ylim),llcrnrlat=ylim[0],urcrnrlat=ylim[1],llcrnrlon=xlim[0],urcrnrlon=xlim[1],resolution='i')
m.drawcoastlines(); m.drawstates(), m.drawcountries()
cs = m.contourf(lon,lat,z_corrected,clevs,cmap='turbo',extend='both')
cs2 = m.plot(lon[:,0]+0.03,lat[:,0]-0.03,'--k',zorder=4)
cs3 = m.plot(lon[:,-1]-0.03,lat[:,-1]+0.03,'--k',zorder=4)
parallels = np.arange(-90,90,step = 2)
m.drawparallels(parallels, labels = [True, False, False, False], fontsize = 22)

meridians = np.arange(0,360, step = 2)
m.drawmeridians(meridians, labels = [False, False, False, True], fontsize = 22)

cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize = label_size)
cbar.set_label(label = '[dBZ]',size = label_size)
plt.title(r'$\bf{b)}$  Near-Surface KuPR 09/17/2017 0601 UTC', size = 26, y = 1.02)
#Add the annuli
circle = Circle(xy = m(storm_lon,storm_lat), radius = 0.45045, fill = False, linewidth = 2)
plt.gca().add_patch(circle)

circle = Circle(xy = m(storm_lon,storm_lat), radius = 1.351351, fill = False, linewidth = 2)
plt.gca().add_patch(circle)

x2star,y2star = m(storm_lon,storm_lat)
#m.plot(x2star,y2star,markersize=26, color = 'k')
m.scatter(x2star,y2star,s=200, color = 'k')

#Plot arrow

arrow_length_x = dx
arrow_length_y = dy


x,y = m(storm_lon,storm_lat)
x2,y2 = m(dx,dy)
plt.arrow(x,y,x2,y2, width = 0.008, head_width = 0.2, head_length = 0.2, color = 'k')

x3,y3 = lons, (perp_slope)*lons + intercept
m.plot(x3,y3,linewidth = 2, color = 'k')

#Now draw another line extending down from the storm_lon, storm_lat point

x4,y4 = coords_lon,coords_lat
m.plot(x4,y4, linewidth = 2, color = 'k')


label_dr = 'UR'
x_dr,y_dr = -110.5,20.9
plt.text(x_dr,y_dr,label_dr, weight = 'bold',size = 24)

label_dr = 'UL'
x_dr,y_dr = -109.7,20.65
plt.text(x_dr,y_dr,label_dr, weight = 'bold',size = 24)



label_dr = 'DR'
x_dr,y_dr = -111.1,19.7
plt.text(x_dr,y_dr,label_dr, weight = 'bold',size = 24)

label_dr = 'DL'
x_dr,y_dr = -110.2,19.5
plt.text(x_dr,y_dr,label_dr, weight = 'bold',size = 24)




plt.show()