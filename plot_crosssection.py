import cartopy.crs as ccrs
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as mpl_cm

import thermodynamics as th
from iris_read import *
from plot_profile_from_UKV import convert_to_ukv_coords, index_selector
from plot_xsect import get_grid_latlon_from_rotated, add_grid_latlon_to_cube, get_coord_index
from sonde_locs import sonde_locs
from thermodynamics import potential_temperature

def centred_cnorm(cube):
    w_min = cube.data.min()
    w_max = cube.data.max()
    range_max = max(abs(w_max), abs(w_min))
    return colors.Normalize(vmin=-range_max, vmax=range_max)


if __name__ == '__main__':

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
    p_cube = read_variable(filename, 408, h)
    q_cube = read_variable(filename, 10, h)

    # check level heights
    if q_cube.coord('level_height').points[0] == t_cube.coord('level_height').points[0]:
        pass
    elif q_cube.coord('level_height').points[1] == t_cube.coord('level_height').points[0]:
        q_cube = q_cube[1:]
    else:
        raise ValueError('Double check the T and q level_heights - they do not match')

    # add true lat lon
    grid_latlon = get_grid_latlon_from_rotated(w_cube)
    add_grid_latlon_to_cube(w_cube, grid_latlon)
    add_grid_latlon_to_cube(p_cube, grid_latlon)

    # convert to theta and make cube
    # this ignores things like the stash code, but is probably fine for now
    # can also convert whole cubes, but to save time/memory only convert slices
    theta = potential_temperature(t_cube.data, p_cube.data)
    theta_cube = p_cube.copy()
    theta_cube.data = theta
    theta_cube.units = 'K'
    theta_cube.standard_name = 'air_potential_temperature'

    # calculate RH
    RH = th.q_p_to_e(q_cube.data, p_cube.data) / th.esat(t_cube.data)
    RH_cube = p_cube.copy()
    RH_cube.data = RH
    RH_cube.standard_name = 'relative_humidity'
    RH_cube.units = '1'

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
    # this could be replaced by built-in function nearest_neighbour_index
    lat_index = index_selector(model_lat, grid_lats)
    lon_index_west = index_selector(model_westbound, grid_lons)
    lon_index_east = index_selector(model_eastbound, grid_lons)

    cmap = mpl_cm.get_cmap("brewer_PuOr_11")

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


    w_con = iplt.contourf(w_single_level, coords=['longitude', 'latitude'],
                          cmap=cmap, norm=centred_cnorm(w_single_level))
    ax.plot(grid_latlon['true_lons'][lat_index, lon_index_west:lon_index_east+1],
             grid_latlon['true_lats'][lat_index, lon_index_west:lon_index_east+1],
             color='k', zorder=50)
    plt.scatter(*sonde_locs['valentia'], marker='*', color='r', edgecolors='k', s=250, zorder=100)

    ax.gridlines(crs=crs_latlon, draw_labels=True)
    ax.set_xlabel('True Longitude / deg')
    ax.set_ylabel('True Latitude / deg')
    plt.colorbar(w_con, label='Upward air velocity / m/s',
                 location='bottom',
                 # orientation='vertical'
                 )
    plt.title(f'UKV {w_cube.coord("level_height").points[height_index]:.0f} '
              f'm {year}/{month}/{day} at {h}h ({forecast_time})')

    plt.tight_layout()
    plt.savefig(f'plots/xsect_map_lat{lat}_{year}{month}{day}_{h}.png', dpi=300)
    plt.show()

    plt.figure()
    # max height
    max_height = 5000
    w_height_mask = (w_cube.coord('level_height').points < max_height)
    t_height_mask = (t_cube.coord('level_height').points < max_height)

    # currently plots cross-section along a model latitude!
    # this is not the same as a true latitude (even though that is displayed on the axis)!
    w_section = w_cube[w_height_mask, lat_index, lon_index_west: lon_index_east + 1]
    theta_section = theta_cube[t_height_mask, lat_index, lon_index_west: lon_index_east + 1]
    RH_section = RH_cube[t_height_mask, lat_index, lon_index_west: lon_index_east + 1]

    w_con = iplt.contourf(w_section, coords=['longitude', 'level_height'],
                          cmap=cmap, norm=centred_cnorm(w_section))

    theta_con = iplt.contour(theta_section, coords=['longitude', 'level_height'],
                             colors='k', linestyles='--')

    RH_con = iplt.contour(RH_section, levels=[0.75], coords=['longitude', 'level_height'],
                             colors='gray', linestyles='-.')
    plt.clabel(theta_con)

    plt.xlabel('True Longitude / deg')
    plt.ylabel('Level height / m')
    plt.title(f'Cross-section approximately along lat {lat} deg')
    plt.colorbar(w_con, label='Upward air velocity / m/s')
    plt.tight_layout()
    plt.savefig(f'plots/xsect_lat{lat}_{year}{month}{day}_{h}.png', dpi=300)
    plt.show()