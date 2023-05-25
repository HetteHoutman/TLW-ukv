import cartopy.crs as ccrs
import iris.plot as iplt
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt

import thermodynamics as th
from cube_processing import cube_at_single_level, check_level_heights, cube_slice, cube_from_array_and_cube, \
    great_circle_xsect, add_orography
from general_plotting_fns import centred_cnorm
from iris_read import *
from miscellaneous import make_great_circle_points, make_great_circle_iris_traj
from plot_profile_from_UKV import convert_to_ukv_coords
from plot_xsect import get_grid_latlon_from_rotated, add_grid_latlon_to_cube
from pp_processing import data_from_pp_filename
from sonde_locs import sonde_locs
from thermodynamics import potential_temperature


def plot_xsect_latitude(w_section, theta_section, RH_section, orog_section, max_height=5000, cmap=mpl_cm.get_cmap("brewer_PuOr_11"),
               coords=None):
    """plots the latitudinal cross-section with filled contours of w and normal contours of theta and RH"""
    if coords is None:
        coords = ['longitude', 'altitude']

    plt.figure()

    w_con = iplt.contourf(w_section, coords=coords,
                          cmap=cmap, norm=centred_cnorm(w_section.data))
    theta_con = iplt.contour(theta_section, coords=coords,
                             colors='k', linestyles='--')
    RH_con = iplt.contour(RH_section, levels=[0.75], coords=coords,
                          colors='0.5', linestyles='-.')

    iplt.plot(orog_section.coord('longitude'), orog_section, color='k')
    plt.fill_between(orog_section.coord('longitude').points, orog_section.data, where=orog_section.data>0, color='k',
                     interpolate=True)

    lat = w_section.coord('latitude').points[0]
    plt.ylim((0,max_height))
    plt.clabel(theta_con)
    plt.xlabel(f'{w_section.coord(coords[0]).name().capitalize()} / deg')
    plt.ylabel(f'{w_section.coord(coords[1]).name().capitalize()} / {str(w_section.coord(coords[1]).units)}')
    plt.title(f'Cross-section approximately along lat {lat} deg')
    plt.colorbar(w_con, label='Upward air velocity / m/s')

    plt.tight_layout()
    plt.savefig(f'plots/xsect_lat{lat:.3f}_{year}{month}{day}_{h}.png', dpi=300)
    plt.show()


def plot_xsect_map(cube_single_level, great_circle=None, cmap=mpl_cm.get_cmap("brewer_PuOr_11"), custom_save=''):
    """
    Plots the map indicating the cross-section, in addition to the w field.
    Parameters
    ----------
    cube_single_level : Cube
        the single level cube to be plotted (can only have height coordinate as aux coord)
    cmap :
        colors
    end : tuple
        lon/lat of the end of the great circle cross-section line to be plotted, or None if no line is to be plotted
    start : tuple
        lon/lat of the start of the great circle cross-section line to be plotted, or None if no line is to be plotted
    custom_save : str
        optional addition to the save file name to distinguish it from others
    """

    crs_latlon = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_latlon})
    ax.coastlines()

    w_con = iplt.contourf(cube_single_level, coords=['longitude', 'latitude'],
                          cmap=cmap, norm=centred_cnorm(cube_single_level.data))

    if great_circle is not None:
        ax.plot(great_circle[0], great_circle[1], color='k', zorder=50)

    plt.scatter(*sonde_locs['valentia'], marker='*', color='r', edgecolors='k', s=250, zorder=100)

    ax.gridlines(crs=crs_latlon, draw_labels=True)
    ax.set_xlabel('True Longitude / deg')
    ax.set_ylabel('True Latitude / deg')
    plt.colorbar(w_con, label='Upward air velocity / m/s',
                 location='bottom',
                 # orientation='vertical'
                 )
    plt.title(f'UKV {cube_single_level.coord("level_height").points[0]:.0f} '
              f'm {year}/{month}/{day} at {h}h ({forecast_time})')

    plt.tight_layout()
    plt.savefig(f'plots/xsect_map{custom_save}_{year}{month}{day}_{h}.png', dpi=300)
    plt.show()


def plot_interpolated_xsect(w_xsect, theta_xsect, RH_xsect, w_y, theta_y, great_circle,
                            cmap=mpl_cm.get_cmap("brewer_PuOr_11")):
    """
    Plots the interpolated cross-section
    Parameters
    ----------
    great_circle
    theta_y : ndarray
        y coordinates belonging to theta_xsect
    w_y : ndarray
        y coordinates belonging to w_xsect
    theta_xsect
    RH_xsect
    w_xsect
    cmap

    Returns
    -------

    """
    x_coords = great_circle[0]

    w_con = plt.contourf(x_coords, w_y, w_xsect, norm=centred_cnorm(w_xsect), cmap=cmap)
    theta_con = plt.contour(x_coords, theta_y, theta_xsect,
                             colors='k', linestyles='--')
    RH_con = plt.contour(x_coords, theta_y, RH_xsect, levels=[0.75],
                          colors='0.5', linestyles='-.')
    plt.savefig('plots/interpolated_xsect_test.png', dpi=300)
    plt.show()


def load_and_process(reg_filename, orog_filename):
    w_cube = read_variable(reg_filename, 150, h)
    t_cube = read_variable(reg_filename, 16004, h)
    p_cube = read_variable(reg_filename, 408, h)
    q_cube = read_variable(reg_filename, 10, h)
    orog_cube = read_variable(orog_filename, 33, 9)

    # check level heights
    q_cube = check_level_heights(q_cube, t_cube)

    # add true lat lon
    grid_latlon = get_grid_latlon_from_rotated(w_cube)
    add_grid_latlon_to_cube(w_cube, grid_latlon)
    add_grid_latlon_to_cube(p_cube, grid_latlon)
    add_grid_latlon_to_cube(orog_cube, grid_latlon)

    # create theta and RH cubes
    theta_cube = cube_from_array_and_cube(potential_temperature(t_cube.data, p_cube.data), p_cube, unit='K',
                                          std_name='air_potential_temperature')
    RH_cube = cube_from_array_and_cube(th.q_p_to_e(q_cube.data, p_cube.data) / th.esat(t_cube.data), p_cube, unit='1',
                                       std_name='relative_humidity')

    # now add orography hybrid height factory to desired cubes
    w_cube, theta_cube, RH_cube = add_orography(orog_cube, w_cube, theta_cube, RH_cube)

    return w_cube, theta_cube, RH_cube, orog_cube


if __name__ == '__main__':

    indir = '/home/users/sw825517/Documents/ukv_data/'
    reg_file = indir + 'prodm_op_ukv_20150414_09_004.pp'
    orog_file = indir + 'prods_op_ukv_20150414_09_000.pp'

    h = 12

    map_bottomleft = (-9.6, 51.6)
    map_topright = (-8.9, 52.1)
    map_height = 750

    max_height = 5000

    gc_start = (-9.5, 51.8)
    gc_end = (-9, 52)

    year, month, day, forecast_time = data_from_pp_filename(reg_file)

    w_cube, theta_cube, RH_cube, orog_cube = load_and_process(reg_file, orog_file)

    gc = make_great_circle_points(gc_start, gc_end, n=200)

    w_single_level = cube_at_single_level(w_cube, map_height, bottomleft=map_bottomleft, topright=map_topright)
    plot_xsect_map(w_single_level, great_circle=gc)

    # xs_bottomleft = (-10.4, 51.9)
    # xs_topright = (-9.25, 51.9)
    # w_section = cube_slice(w_cube, xs_bottomleft, xs_topright, height=(0, max_height), force_latitude=True)
    # theta_section = cube_slice(theta_cube, xs_bottomleft, xs_topright, height=(0, max_height), force_latitude=True)
    # RH_section = cube_slice(RH_cube, xs_bottomleft, xs_topright, height=(0, max_height), force_latitude=True)
    # orog_section = cube_slice(orog_cube, xs_bottomleft, xs_topright, force_latitude=True)
    # plot_xsect_latitude(w_section, theta_section, RH_section, orog_section)

    w_sliced = cube_slice(w_cube, map_bottomleft, map_topright, height=(0, max_height))
    theta_sliced = cube_slice(theta_cube, map_bottomleft, map_topright, height=(0, max_height))
    RH_sliced = cube_slice(RH_cube, map_bottomleft, map_topright, height=(0, max_height))
    # w_xsect = great_circle_xsect(w_sliced, gc, n=200)
    # theta_xsect = great_circle_xsect(theta_sliced, gc, n=200)
    # RH_xsect = great_circle_xsect(RH_sliced, gc, n=200)
    # orog_xsect = great_circle_xsect(w_sliced, gc_start=gc_start, gc_end=gc_end, n=200)
    #
    # plot_interpolated_xsect(w_xsect, theta_xsect, RH_xsect, w_sliced.coord('level_height').points,
    #                         theta_sliced.coord('level_height').points, gc)

    crs_latlon = ccrs.PlateCarree()
    crs_rotated = w_cube.coord('grid_latitude').coord_system.as_cartopy_crs()
    gc_model = np.array(
        [convert_to_ukv_coords(coords[0], coords[1], crs_latlon, crs_rotated) for coords in gc.T])

    traj = make_great_circle_iris_traj(gc_model, 200)
    result = traj.interpolate(w_sliced, method='linear')
    iplt.contourf(result)
    plt.show()

    # TODO so somehow this thinks i'm giving






    # w_slice = np.empty((max_height_index + 1, n))
    # start_time = time.clock()
    # for k in range(max_height_index + 1):
    #     w_slice[k] = scipy.interpolate.griddata(points, w[k].data[::-1].flatten(), gc_model)
    #     if k % 5 == 0:
    #         print(f'at index {k}/{max_height_index}...')
    # print(f'{time.clock()-start_time} seconds needed to iterate over height and fill w_slice')

    # TODO ask Sue for her oblique xsect code to compare
    """def plot(bl, tr):
    ...:     new_crs = ccrs.RotatedPole(pole_latitude=bl[1], pole_longitude=bl[0])
    ...:     rot = new_crs.transform_point(*tr, crs_latlon)[0]
    ...:     new_crs = ccrs.RotatedPole(pole_latitude=bl[1], pole_longitude=bl[0], central_rotated_longitude=-rot)
    ...:     iplt.contourf(w[0])
    ...:     ax = plt.gca()
    ...:     ax.coastlines()
    ...:     ax.gridlines(crs=new_crs)
    ...:     gc = np.array(g.npts(*bl, *tr, n)).T
    ...:     plt.plot(gc[0], gc[1], transform=crs_latlon)
    ...:     plt.show()
    ...:     return new_crs
    
    so can construct a projection that has great circle as a longitude line.
    now need to figure out how to construct grid on this projection and then regrid w onto this new grid??
"""