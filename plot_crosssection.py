import cartopy.crs as ccrs
import iris.plot as iplt
import matplotlib.pyplot as plt
from iris_read import *

from plot_profile_from_UKV import convert_to_ukv_coords, index_selector
from plot_xsect import get_grid_latlon_from_rotated, add_grid_latlon_to_cube
from sonde_locs import sonde_locs

# read file
indir = '/home/users/sw825517/Documents/ukv_data/'
filename = indir + 'prodm_op_ukv_20150414_09_004.pp'

year = filename[-18:-14]
month = filename[-14:-12]
day = filename[-12:-10]
forecast_time = filename[-9:-7]
h = 12

w_cube = read_variable(filename, 150, h)
t_cube = read_variable(filename, 16004, h)
grid_latlon = get_grid_latlon_from_rotated(w_cube)
add_grid_latlon_to_cube(w_cube, grid_latlon)

# here choose cross-section details (only supports along latitude for now)
lat = 51.9
lonbound_west = -10.5
lonbound_east = -8.5

grid_lats = w_cube.coord('grid_latitude').points
grid_lons = w_cube.coord('grid_longitude').points

crs_latlon = ccrs.PlateCarree()
crs_rotated = w_cube.coord('grid_latitude').coord_system.as_cartopy_crs()

model_westbound, model_lat = convert_to_ukv_coords(lonbound_west, lat, crs_latlon, crs_rotated)
model_eastbound, temp = convert_to_ukv_coords(lonbound_east, lat, crs_latlon, crs_rotated)
lat_index = index_selector(model_lat, grid_lats)
lon_index_west = index_selector(model_westbound, grid_lons)
lon_index_east = index_selector(model_eastbound, grid_lons)

# max height
height_mask = (w_cube.coord('level_height').points < 5000)

# currently plots cross section along a model latitude!
# this is not the same as a true latitude (even though that is displayed on the axis)!
w_section = w_cube[height_mask, lat_index, lon_index_west: lon_index_east + 1]
con = iplt.contourf(w_section, coords=['longitude', 'level_height'])
plt.xlabel('True Longitude / deg')
plt.ylabel('Level height / m')
plt.title(f'Cross-section approximately along lat {lat} deg')
plt.colorbar(con, label='Upward air velocity / m/s')
plt.show()

fig, ax = plt.subplots(1,1, subplot_kw={'projection': crs_latlon})
ax.coastlines()

map_height = 1000
height_index = w_cube.coord('level_height').nearest_neighbour_index(map_height)
bl = (-10.5, 50.5)
tr = (-5, 56)

bl_model = crs_rotated.transform_point(bl[0], bl[1], crs_latlon)
tr_model = crs_rotated.transform_point(tr[0], tr[1], crs_latlon)

w_single_level = w_cube[1].intersection(grid_latitude=(bl_model[1], tr_model[1]),
                                        grid_longitude=(bl_model[0], tr_model[0]))

con = iplt.contourf(w_single_level, coords=['longitude', 'latitude'])
ax.plot(grid_latlon['true_lons'][lat_index, lon_index_west:lon_index_east+1],
         grid_latlon['true_lats'][lat_index, lon_index_west:lon_index_east+1],
         color='yellow', zorder=50)
plt.scatter(*sonde_locs['valentia'], marker='*', color='red', s=100, zorder=100)

ax.gridlines(crs=crs_latlon, draw_labels=True)
ax.set_xlabel('True Longitude / deg')
ax.set_ylabel('True Latitude / deg')
plt.colorbar(con, label='Upward air velocity / m/s', location='bottom')
plt.title(f'UKV {w_cube.coord("level_height").points[height_index]:.0f} '
          f'm {year}/{month}/{day} at {h}h ({forecast_time})')

plt.tight_layout()
plt.show()