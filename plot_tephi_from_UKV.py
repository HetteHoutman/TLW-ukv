import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import tephi

import thermodynamics as th
from cube_processing import check_level_heights, read_variable
from met_fns import uv_to_spddir
from miscellaneous import convert_to_ukv_coords, index_selector

indir = '/home/users/sw825517/Documents/ukv_data/'
filename = indir + 'prodm_op_ukv_20150414_09_004.pp'

year = filename[-18:-14]
month = filename[-14:-12]
day = filename[-12:-10]
forecast_time = filename[-9:-7]
h = 12

u_cube = read_variable(filename, 2, h)
v_cube = read_variable(filename, 3, h)
p_theta_cube = read_variable(filename, 408, h)
T_cube = read_variable(filename, 16004, h)
q_cube = read_variable(filename, 10, h)

q_cube = check_level_heights(q_cube, T_cube)

# only plot up to certain T_height
min_height = 0
max_height = 10000
T_height = T_cube.coord('level_height').points
rho_height = u_cube.coord('level_height').points
T_mask = (T_height < max_height) & (T_height > min_height)
rho_mask = (rho_height < max_height) & (rho_height > min_height)

# coordinates
xpos = -10.35
ypos = 51.9

lats = T_cube.coord('grid_latitude').points
lons = T_cube.coord('grid_longitude').points

crs_latlon = ccrs.PlateCarree()
crs_rotated = u_cube.coord('grid_latitude').coord_system.as_cartopy_crs()

model_x, model_y = convert_to_ukv_coords(xpos, ypos, crs_latlon, crs_rotated)

lat_index = index_selector(model_y, lats)
lon_index = index_selector(model_x, lons)
true_model_x = lons[lon_index]
true_model_y = lats[lat_index]

# extract columns at appropriate location
pt_col = p_theta_cube.data[T_mask, lat_index, lon_index]
T_col = T_cube.data[T_mask, lat_index, lon_index]
q_col = q_cube.data[T_mask, lat_index, lon_index]

# interpolate winds as we're Arakawa C-grid
u_col = 0.5*(u_cube.data[rho_mask, lat_index, lon_index-1] +
             u_cube.data[rho_mask, lat_index, lon_index])
v_col = 0.5*(v_cube.data[rho_mask, lat_index-1, lon_index] +
             v_cube.data[rho_mask, lat_index, lon_index])
spd_col, dir_col = uv_to_spddir(u_col, v_col)

dew_col = th.dewpoint(T_col, pt_col, q_col)

tephi.MIN_THETA = -40
tpg = tephi.Tephigram(anchor=[(1050, -30), (200, -30)])
temp_line = tpg.plot(zip(pt_col / 100, T_col - 273.15))
tpg.plot(zip(pt_col / 100, dew_col - 273.15))

true_x, true_y = crs_latlon.transform_point(true_model_x, true_model_y, crs_rotated)
title = f'UKV ({true_x:.02f}, {true_y:.02f}) on {year}/{month}/{day} at {h} ({forecast_time})'
plt.suptitle(title)
plt.tight_layout()
plt.savefig(f'plots/tephi_from_UKV_({true_x:.02f}_{true_y:.02f})_{year}{month}{day}_{h}.png', dpi=300)
plt.show()