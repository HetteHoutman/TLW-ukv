import matplotlib.pyplot as plt
import iris
import iris.quickplot as qplt
import numpy as np

from iris_read import *
from plot_profile_from_UKV import latlon_index_selector, convert_to_ukv_coords, uv_to_spddir
import cartopy.crs as ccrs
import tephi
import thermodynamics as th

indir = '/home/users/sw825517/Documents/ukv_data/'
filename = indir + 'prodm_op_ukv_20150414_09_004.pp'

h = 12

p_rho_cube = read_variable(filename, 407, h)
u_cube = read_variable(filename, 2, h)
v_cube = read_variable(filename, 3, h)
p_theta_cube = read_variable(filename, 408, h)
T_cube = read_variable(filename, 16004, h)
q_cube = read_variable(filename, 10, h)

# only plot up to certain height
min_height = 20
max_height = 5000
height = u_cube.coord('level_height').points
level_mask = (height < max_height) & (height > min_height)
height = height[level_mask]

# coordinates
xpos = -10.35
ypos = 51.9

lats = u_cube.coord('grid_latitude').points
lons = u_cube.coord('grid_longitude').points

crs_latlon = ccrs.PlateCarree()
crs_rotated = u_cube.coord('grid_latitude').coord_system.as_cartopy_crs()

model_x, model_y = convert_to_ukv_coords(xpos, ypos, crs_latlon, crs_rotated)

lat_index, lon_index = latlon_index_selector(model_y, model_x, lats, lons)
true_model_x = lons[lon_index]
true_model_y = lats[lat_index]

# extract columns at appropriate location
"""cant use level_mask because levels differ, need to find a way around"""
pr_col = p_rho_cube.data[level_mask, lat_index, lon_index]
pt_col = p_theta_cube.data[level_mask, lat_index, lon_index]
T_col = T_cube.data[level_mask, lat_index, lon_index]
q_col = q_cube.data[level_mask, lat_index, lon_index]
u_col = u_cube.data[level_mask, lat_index, lon_index]
v_col = v_cube.data[level_mask, lat_index, lon_index]

spd_col, dir_col = uv_to_spddir(u_col, v_col)
dew_col = th.dewpoint(T_col, pt_col, q_col)

tephi.MIN_THETA = -40
tpg = tephi.Tephigram(anchor=[(1050, -40), (200, -40)])
# so use read_variable function and thermodynamics to calculate dewpoint
# however, q and t have different model levels, so need to interpolate
# don't know how p clarks code does this, and i cant get interpolate to work...
# try again with clearer mind