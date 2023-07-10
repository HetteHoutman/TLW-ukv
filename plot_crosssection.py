import sys

import iris.coords
import iris.plot as iplt
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import os

import thermodynamics as th
from cube_processing import *
from miscellaneous import make_great_circle_points, convert_list_to_ukv_coords, check_argv_num, load_settings
from plotting_fns import centred_cnorm
from pp_processing import data_from_pp_filename
from sonde_locs import sonde_locs


# keep in mind that  the functions here might use global variables specific to this file

def plot_xsect_map(cube_single_level, great_circle=None, cmap="brewer_PuOr_11", custom_save='',
                   whitespace=True):
    """
    Plots the map indicating the cross-section, in addition to the w field.
    Parameters
    ----------
    cube_single_level : Cube
        the single level cube to be plotted (can only have height coordinate as aux coord)
    great_circle : ndarray
        of shape (2, n). lon/lat pairs of points on great circle
    cmap :
        colors
    custom_save : str
        optional addition to the save file name to distinguish it from others
    whitespace : bool
        if False, sets ylim and xlim such that the whitespace caused by reprojection from ukv grid is not shown.
        this cuts off some of the data from the plot, too
    """

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_latlon})
    ax.coastlines()

    w_con = iplt.contourf(cube_single_level, coords=['longitude', 'latitude'],
                          cmap=mpl_cm.get_cmap(cmap), norm=centred_cnorm(cube_single_level.data))

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

    if not whitespace:
        lats = cube_single_level.coord('latitude').points
        lons = cube_single_level.coord('longitude').points
        plt.xlim(lons[0, 0], lons[-1, -1])
        plt.ylim(lats[0, -1], lats[-1, 0])

    plt.title(f'UKV {s.map_height:.0f} '
              f'm {year}/{month}/{day} at {s.h}h ({forecast_time})')

    plt.tight_layout()
    my_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(my_dir, f'plots/xsect_map{custom_save}_{year}{month}{day}_{s.h}.png'), dpi=300)
    plt.show()


def plot_xsect(w_xsect, theta_xsect, RH_xsect, max_height=5000, cmap="brewer_PuOr_11", custom_save='', RH_level=0.75,
               figsize=None):
    """
    Plots the cross section of the w, theta and RH fields.
    Parameters
    ----------
    w_xsect : Cube
        Cube containing the w field interpolated along line using trajectory method
    theta_xsect : Cube
        Cube containing the theta field interpolated along line using trajectory method
    RH_xsect : Cube
        Cube containing the RH field interpolated along line using trajectory method
    max_height : float or int
        altitude(!) at which to cut off plot
    cmap : colormap
        colormap for the plot
    custom_save : str
        optional addition to the save file name to distinguish it from others
    RH_level : float
        relative humidity above which to hatch
    figsize : tuple
        figure size

    Returns
    -------

    """
    coords = ['distance_from_start', 'altitude']
    if figsize is not None:
        plt.figure(figsize=figsize)
    w_con = iplt.contourf(w_xsect, cmap=mpl_cm.get_cmap(cmap), norm=centred_cnorm(w_xsect.data), coords=coords)
    theta_con = iplt.contour(theta_xsect, colors='k', linestyles='--', coords=coords)
    RH_con = iplt.contourf(RH_xsect, levels=[RH_level, 1], colors='none', linestyles='none', coords=coords, hatches=['..'])

    orog = w_xsect.coord('surface_altitude').points
    x = w_xsect.coord('distance_from_start').points
    plt.fill_between(x, orog, where=orog > 0, color='k', interpolate=True)

    plt.colorbar(w_con, label='Upward air velocity / m/s',
                 # location='bottom',
                 # orientation='vertical'
                 )
    plt.clabel(theta_con)

    plt.ylabel('Altitude / m')
    plt.ylim((0, max_height))
    plt.xlabel(f'Distance along great circle / {w_xsect.coord(coords[0]).units}')

    my_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(my_dir, f'plots/xsect{custom_save}_{year}{month}{day}_{s.h}.png'), dpi=300)
    plt.tight_layout()


def load_and_process(reg_filename, orog_filename):
    w_cube = read_variable(reg_filename, 150, s.h)
    t_cube = read_variable(reg_filename, 16004, s.h)
    p_cube = read_variable(reg_filename, 408, s.h)
    q_cube = read_variable(reg_filename, 10, s.h)
    orog_cube = read_variable(orog_filename, 33, s.orog_h)

    # check level heights
    q_cube = check_level_heights(q_cube, t_cube)

    # add true lat lon
    add_true_latlon_coords(w_cube, p_cube, orog_cube)

    # create theta and RH cubes
    theta_cube = cube_from_array_and_cube(th.potential_temperature(t_cube.data, p_cube.data), p_cube, unit='K',
                                          std_name='air_potential_temperature')
    RH_cube = cube_from_array_and_cube(th.q_p_to_e(q_cube.data, p_cube.data) / th.esat(t_cube.data), p_cube, unit='1',
                                       std_name='relative_humidity')

    # now add orography hybrid height factory to desired cubes
    add_orography(orog_cube, w_cube, theta_cube, RH_cube)

    return w_cube, theta_cube, RH_cube


if __name__ == '__main__':
    # check correct number of cmd arguments
    check_argv_num(sys.argv, 1, message='(settings json file)')

    # load settings
    s = load_settings(sys.argv[1])

    # load cubes
    year, month, day, forecast_time = data_from_pp_filename(s.reg_file)
    cubes = load_and_process(s.reg_file, s.orog_file)

    # define coordinate systems
    crs_latlon = ccrs.PlateCarree()
    crs_rotated = cubes[0].coord('grid_latitude').coord_system.as_cartopy_crs()

    # make great circle for interpolation
    gc, dists = make_great_circle_points(s.gc_start, s.gc_end, n=s.n)
    gc_model = convert_list_to_ukv_coords(gc[0], gc[1], crs_latlon, crs_rotated)

    # plot map for clarity
    w_single_level = cube_at_single_level(cubes[0], s.map_height, coord='altitude', bottomleft=s.map_bottomleft, topright=s.map_topright)
    plot_xsect_map(w_single_level[0], great_circle=gc, whitespace=True)

    # make cross-sections from sliced cubed to save computation time and plot them
    cubes_sliced = cube_slice(*cubes, bottom_left=s.interp_bottomleft, top_right=s.interp_topright,
                              height=(0, s.max_height))
    cubes_xsect = cube_custom_line_interpolate(gc_model, *cubes_sliced)
    add_dist_coord(dists, *cubes_xsect)

    plot_xsect(*cubes_xsect, max_height=s.max_height, RH_level=s.RH_level, figsize=(8, 4))
    plt.show()
