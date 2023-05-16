import cartopy.crs as ccrs
import iris.coords
import iris.plot as iplt
import matplotlib.cm as mpl_cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import thermodynamics as th
from iris_read import *
from plot_profile_from_UKV import convert_to_ukv_coords
from plot_xsect import get_grid_latlon_from_rotated, add_grid_latlon_to_cube, get_coord_index
from sonde_locs import sonde_locs
from thermodynamics import potential_temperature


def centred_cnorm(cube):
    w_min = cube.data.min()
    w_max = cube.data.max()
    range_max = max(abs(w_max), abs(w_min))
    return colors.Normalize(vmin=-range_max, vmax=range_max)

def check_level_heights(q, t):
    """check whether q and Temperature cubes have same level heights and adjust if necessary."""
    if q.coord('level_height').points[0] == t.coord('level_height').points[0]:
        pass
    elif q.coord('level_height').points[1] == t.coord('level_height').points[0]:
        q = q[1:]
    else:
        raise ValueError('Double check the T and q level_heights - they do not match')
    return q


def plot_xsect(w, theta, RH, max_height=5000, cmap=mpl_cm.get_cmap("brewer_PuOr_11"),
               coords=None):
    """plots the cross-section with filled contours of w and normal contours of theta and RH"""
    if coords is None:
        coords = ['longitude', 'altitude']

    plt.figure()

    # if vertical coordinate is 3-dimensional (such as with altitude), need to select slice, else not
    # if w_cube.coord(coords[1]).ndim == 3:
    #     w_height_mask = (w.coord(coords[1]).points[:, lat_index, lon_index_west:lon_index_east+1] < max_height)
    #     t_height_mask = (theta.coord(coords[1]).points[:, lat_index, lon_index_west:lon_index_east+1] < max_height)
    # else:
    #     w_height_mask = (w.coord(coords[1]).points < max_height)
    #     t_height_mask = (theta.coord(coords[1]).points < max_height)

    w_height_mask = (w.coord('level_height').points < max_height)
    t_height_mask = (theta.coord('level_height').points < max_height)


    # currently plots cross-section along a model latitude!
    # this is not the same as a true latitude (even though that is displayed on the axis)!
    # NB uses lat index etc as global variables!!
    w_section = w[w_height_mask, lat_index, lon_index_west: lon_index_east + 1]
    theta_section = theta[t_height_mask, lat_index, lon_index_west: lon_index_east + 1]
    RH_section = RH[t_height_mask, lat_index, lon_index_west: lon_index_east + 1]

    w_con = iplt.contourf(w_section, coords=coords,
                          cmap=cmap, norm=centred_cnorm(w_section))
    theta_con = iplt.contour(theta_section, coords=coords,
                             colors='k', linestyles='--')
    RH_con = iplt.contour(RH_section, levels=[0.75], coords=coords,
                          colors='gray', linestyles='-.')

    plt.clabel(theta_con)
    plt.xlabel(f'{w_section.coord(coords[0]).name().capitalize()} / deg')
    plt.ylabel(f'{w_section.coord(coords[1]).name().capitalize()} / {str(w_section.coord(coords[1]).units)}')
    plt.title(f'Cross-section approximately along lat {lat} deg')
    plt.colorbar(w_con, label='Upward air velocity / m/s')

    plt.tight_layout()
    plt.savefig(f'plots/xsect_lat{lat}_{year}{month}{day}_{h}.png', dpi=300)
    plt.show()


def plot_xsect_map(w, map_height=1000, cmap=mpl_cm.get_cmap("brewer_PuOr_11"),
                   bottomleft=(-10.5, 50.5), topright=(-5, 56)):
    """
    Plots the map indicating the cross-section, in addition to the w field.
    Parameters
    ----------
    w
    map_height
    cmap
    bottomleft : tuple
        lon/lat for the bottom left point of the map
    topright : tuple
        lon/lat for the top right point of the map
    """
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_latlon})
    ax.coastlines()

    height_index = w.coord('level_height').nearest_neighbour_index(map_height)

    bl_model = crs_rotated.transform_point(bottomleft[0], bottomleft[1], crs_latlon)
    tr_model = crs_rotated.transform_point(topright[0], topright[1], crs_latlon)

    w_single_level = w[1].intersection(grid_latitude=(bl_model[1], tr_model[1]),
                                          grid_longitude=(bl_model[0], tr_model[0]))
    w_con = iplt.contourf(w_single_level, coords=['longitude', 'latitude'],
                          cmap=cmap, norm=centred_cnorm(w_single_level))

    ax.plot(grid_latlon['true_lons'][lat_index, lon_index_west:lon_index_east + 1],
            grid_latlon['true_lats'][lat_index, lon_index_west:lon_index_east + 1],
            color='k', zorder=50)
    plt.scatter(*sonde_locs['valentia'], marker='*', color='r', edgecolors='k', s=250, zorder=100)

    ax.gridlines(crs=crs_latlon, draw_labels=True)
    ax.set_xlabel('True Longitude / deg')
    ax.set_ylabel('True Latitude / deg')
    plt.colorbar(w_con, label='Upward air velocity / m/s',
                 location='bottom',
                 # orientation='vertical'
                 )
    plt.title(f'UKV {w.coord("level_height").points[height_index]:.0f} '
              f'm {year}/{month}/{day} at {h}h ({forecast_time})')

    plt.tight_layout()
    plt.savefig(f'plots/xsect_map_lat{lat}_{year}{month}{day}_{h}.png', dpi=300)
    plt.show()


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

    q_cube = check_level_heights(q_cube, t_cube)

    orog_file = indir + 'prods_op_ukv_20150414_09_000.pp'
    # NB might be easier not to use read_variable but just iris.load
    orog_cube = read_variable(orog_file, 33, 9)


    # add true lat lon
    grid_latlon = get_grid_latlon_from_rotated(w_cube)
    add_grid_latlon_to_cube(w_cube, grid_latlon)
    add_grid_latlon_to_cube(p_cube, grid_latlon)

    # convert to theta and make cube
    # this ignores things like the stash code, but is probably fine for now
    theta = potential_temperature(t_cube.data, p_cube.data)
    theta_cube = p_cube.copy()
    theta_cube.data = theta
    del theta
    theta_cube.units = 'K'
    theta_cube.standard_name = 'air_potential_temperature'

    # calculate RH
    RH = th.q_p_to_e(q_cube.data, p_cube.data) / th.esat(t_cube.data)
    RH_cube = p_cube.copy()
    RH_cube.data = RH
    del RH, t_cube
    RH_cube.standard_name = 'relative_humidity'
    RH_cube.units = '1'

    # now add orography hybrid height factory to desired cubes
    orog_coord = iris.coords.AuxCoord(orog_cube.data, standard_name=str(orog_cube.standard_name),
                                      long_name='orography', var_name='orog', units=orog_cube.units)
    for cube in [w_cube, theta_cube, RH_cube]:
        sigma = cube.coord('sigma')
        delta = cube.coord('level_height')
        fac = iris.aux_factory.HybridHeightFactory(delta=delta, sigma=sigma, orography=orog_coord)
        cube.add_aux_coord(orog_coord, (get_coord_index(cube, 'grid_latitude'),
                                        get_coord_index(cube, 'grid_longitude')))
        cube.add_aux_factory(fac)
    del orog_coord

    # here choose cross-section details (only supports along latitude for now)
    lat = 51.9
    lonbound_west = -10.5
    lonbound_east = -8.5

    crs_latlon = ccrs.PlateCarree()
    crs_rotated = w_cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    model_westbound, model_lat = convert_to_ukv_coords(lonbound_west, lat, crs_latlon, crs_rotated)
    model_eastbound, temp = convert_to_ukv_coords(lonbound_east, lat, crs_latlon, crs_rotated)
    lat_index = w_cube.coord('grid_latitude').nearest_neighbour_index(model_lat)
    lon_index_west = w_cube.coord('grid_longitude').nearest_neighbour_index(model_westbound)
    lon_index_east = w_cube.coord('grid_longitude').nearest_neighbour_index(model_eastbound)

    plot_xsect_map(w_cube)
    plot_xsect(w_cube, theta_cube, RH_cube)