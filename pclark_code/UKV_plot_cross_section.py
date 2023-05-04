import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import iris

import iris.plot as iplt
import iris.quickplot as qplt

import numpy as np

import thermodynamics as th

from iris_read import *

def xsect_fig(fignum, cross_section, figname, cminmax) :
    '''
    This is just a tarted up contour plot to save repetition.
    '''
    fig = plt.figure(fignum,figsize=(15, 6))
    fig.clf() 

    con=iplt.contourf(cross_section, np.arange(cminmax[0],cminmax[1],1), 
                      coords=['grid_longitude', 'air_pressure'])
    plt.ylim([100000,30000])

# add a colourbar with a label
#    cbar = plt.colorbar(con, colorbar_axes, orientation='horizontal')
    cbar = plt.colorbar(con, orientation='vertical')
    cbar.set_label(T.units)

    plt.savefig(figname)
    plt.close()
 
    return

filename = '/glusterfs/msc/users_2014/rm806551/data/prodm/prodm_op_nae-mn_20120628_00_006_u.pp'
# indir = '/home/users/sw825517/Documents/ukv_data/'
# filename = indir + 'prodm_op_ukv_20230419_12_000.pp'

# Just select out the 06Z data
h = 12
p = read_variable(filename,408,h)
T = read_variable(filename,16004,h)
q = read_variable(filename,10,h)

print(T, p, q)

# Create a new vertical coordinate from the appropriate pressure data.

newcoord = iris.coords.AuxCoord(points=p.data,\
  standard_name=p.standard_name,units=p.units)

#print newcoord.points.shape

#print T.data.shape

# Add the pressure vertical coordinate (3D field) to T and q as aux_coords

T.add_aux_coord(newcoord, [0,1,2])
q.add_aux_coord(newcoord, [0,1,2])

print(T,q)

theta=th.potential_temperature(T.data, p.data)
theta_cube=T.copy()
theta_cube.data=theta
theta_cube.rename('potential temperature')

theta_e=th.equiv_potential_temperature(T.data, p.data, q.data)
theta_e_cube=T.copy()
theta_e_cube.data=theta_e
theta_e_cube.rename('equivalent potential temperature')

theta_es=th.equiv_potential_temperature(T.data, p.data, th.qsat(T.data, p.data))
theta_es_cube=T.copy()
theta_es_cube.data=theta_es
theta_es_cube.rename('saturated equivalent potential temperature')

# This sets up a standard 'Lat/Long' coordinate system
crs_latlon = ccrs.PlateCarree()

# This extracts the rotated lat/long coordinate system from the the cube 
# in the form of a cartopy map projection.
crs_rotated=theta_e_cube[0,:,:].coord('grid_latitude').coord_system.as_cartopy_crs()

fig3 = plt.figure(3,figsize=(15, 10))    
fig3.clf() 

con=qplt.contourf(theta_e_cube[0,:,:],20,hold='on')
plt.gca().coastlines(resolution='50m')

# get the current axes
plt1_ax = plt.gca()

# This lat/long rotated grid - unfortunately draw_labels=True isn't implemented for plotting on rotated grids. 
#plt1_ax.gridlines(crs=crs_latlon, linestyle='-', draw_labels=True)
plt1_ax.gridlines(crs=crs_latlon, linestyle='-')
#plt1_ax.set_xticks(np.arange(-100,110,10),crs=crs_latlon)

#plt1_ax.set_yticks(np.arange(0,90,5),crs=crs_latlon)

# This plots rotated grid - unfortunately draw_labels=True isn't implemented for rotated grids. 
# plt1_ax.gridlines(crs=crs_rotated, draw_labels=True)
plt1_ax.gridlines(crs=crs_rotated)
# Label some gridlines manually
xticks = plt1_ax.get_xticks()
yticks = plt1_ax.get_yticks()
for x in xticks[1:-2]:
    y = yticks[np.floor(np.size(yticks)/2)]
    name=str(x)+','+str(y)
    plt.text(x, y, name, horizontalalignment='center', verticalalignment='center')
for y in yticks[1:-2]:
    x = xticks[np.floor(np.size(xticks)/2)]
    name=str(x)+','+str(y)
    plt.text(x, y, name, horizontalalignment='center', verticalalignment='center')

# Now plot lines corresponding to extracted cross sections.

# Extract arrays of the rotated coordinates.
lats=theta_e_cube.coord('grid_latitude').points
lons=theta_e_cube.coord('grid_longitude').points-360

# Create list of required cross section indices.
lat_indices = range(150, 210, 10)

# Overplot lines on map.
for lat_index in lat_indices:
    plt.plot(lons[[0,-1]],lats[[lat_index,lat_index]])

# Save the map and show it.
plt.savefig('plots/theta_e_map_new.png')
plt.show()

for lat_index in lat_indices:
    indstr='%03d'%lat_index
    xsect_fig(0,theta_cube[:,lat_index,:] , 'plots/Theta_cross_sect_'+indstr+'_new.png',(280,341))

    xsect_fig(1,theta_e_cube[:,lat_index,:] , 'plots/Theta_e_cross_sect_'+indstr+'_new.png',(290,371))

    xsect_fig(2,theta_es_cube[:,lat_index,:] , 'plots/Theta_es_cross_sect_'+indstr+'_new.png',(290,371))

