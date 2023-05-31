import sys
from types import SimpleNamespace

import iris.plot as iplt
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import json

from sonde_locs import sonde_locs
from useful import thermodynamics as th
from useful.cube_processing import *
from useful.general_plotting_fns import centred_cnorm
from useful.miscellaneous import make_great_circle_points, convert_list_to_ukv_coords
from useful.pp_processing import data_from_pp_filename


# keep in mind that  the functions here might use global variables specific to this file
def plot_xsect_latitude(w_section, theta_section, RH_section, orog_section, max_height=5000, cmap=mpl_cm.get_cmap("brewer_PuOr_11"),
               coords=None):
    """
    Not used currently!
    plots the latitudinal cross-section with filled contours of w and normal contours of theta and RH"""
    if coords is None:
        coords = ['longitude', 'altitude']

    plt.figure()

    w_con = iplt.contourf(w_section, coords=coords,
                          cmap=cmap, norm=centred_cnorm(w_section.data))
    theta_con = iplt.contour(theta_section, levels=5, coords=coords,
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
    plt.savefig(f'plots/xsect_lat{lat:.3f}_{year}{month}{day}_{s.h}.png', dpi=300)
    plt.show()


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
        plt.xlim(lons[0,0], lons[-1,-1])
        plt.ylim(lats[0,-1], lats[-1,0])

    plt.title(f'UKV {cube_single_level.coord("level_height").points[0]:.0f} '
              f'm {year}/{month}/{day} at {s.h}h ({forecast_time})')

    plt.tight_layout()
    plt.savefig(f'plots/xsect_map{custom_save}_{year}{month}{day}_{s.h}.png', dpi=300)
    plt.show()


def plot_xsect(w_xsect, theta_xsect, RH_xsect, max_height=5000, cmap="brewer_PuOr_11"):
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

    Returns
    -------

    """
    coords = ['distance_from_start', 'altitude']
    w_con = iplt.contourf(w_xsect, cmap=mpl_cm.get_cmap(cmap), norm=centred_cnorm(w_xsect.data), coords=coords)
    theta_con = iplt.contour(theta_xsect, colors='k', linestyles='--', coords=coords)
    RH_con = iplt.contour(RH_xsect, levels=[0.75], colors='0.5', linestyles='-.', coords=coords)

    orog = w_xsect.coord('surface_altitude').points
    x = w_xsect.coord('distance_from_start').points
    plt.fill_between(x, orog, where=orog>0, color='k', interpolate=True)

    plt.colorbar(w_con, label='Upward air velocity / m/s',
                 # location='bottom',
                 # orientation='vertical'
                 )
    plt.clabel(theta_con)

    plt.ylabel('Altitude / m')
    plt.ylim((0, max_height))
    plt.xlabel('Distance along great circle / m')
    # TODO save to proper name
    plt.savefig('plots/interpolated_xsect_test.png', dpi=300)
    plt.show()

def load_and_process(reg_filename, orog_filename):
    w_cube = read_variable(reg_filename, 150, s.h)
    t_cube = read_variable(reg_filename, 16004, s.h)
    p_cube = read_variable(reg_filename, 408, s.h)
    q_cube = read_variable(reg_filename, 10, s.h)
    orog_cube = read_variable(orog_filename, 33, s.orog_h)

    # check level heights
    q_cube = check_level_heights(q_cube, t_cube)

    # add true lat lon
    grid_latlon = get_grid_latlon_from_rotated(w_cube)
    add_grid_latlon_to_cube(w_cube, grid_latlon)
    add_grid_latlon_to_cube(p_cube, grid_latlon)
    add_grid_latlon_to_cube(orog_cube, grid_latlon)

    # create theta and RH cubes
    theta_cube = cube_from_array_and_cube(th.potential_temperature(t_cube.data, p_cube.data), p_cube, unit='K',
                                          std_name='air_potential_temperature')
    RH_cube = cube_from_array_and_cube(th.q_p_to_e(q_cube.data, p_cube.data) / th.esat(t_cube.data), p_cube, unit='1',
                                       std_name='relative_humidity')

    # now add orography hybrid height factory to desired cubes
    w_cube, theta_cube, RH_cube = add_orography(orog_cube, w_cube, theta_cube, RH_cube)

    return w_cube, theta_cube, RH_cube


if __name__ == '__main__':

    # load settings
    if len(sys.argv) != 2:
        raise Exception(f'Gave {len(sys.argv) - 1} arguments but this file takes exactly 1 (settings.json)')

    file = sys.argv[1]
    with open(file) as f:
        s = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

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
    w_single_level = cube_at_single_level(cubes[0], s.map_height, bottomleft=s.map_bottomleft, topright=s.map_topright)
    plot_xsect_map(w_single_level, great_circle=gc, whitespace=True)

    # make cross-sections and plot them
    cubes_sliced = cube_slice(*cubes, bottom_left=s.interp_bottomleft, top_right=s.interp_topright, height=(0, s.max_height))
    cubes_xsect = cube_custom_line_interpolate(gc_model, *cubes_sliced)
    cubes_xsect = add_dist_coord(dists, *cubes_xsect)

    plot_xsect(*cubes_xsect, max_height=s.max_height)


    # this code below plots a crosssection along a latitude without interpolation
    # can be used to check whether interpolation makes sense

    # xs_bottomleft = (-10.4, 51.9)
    # xs_topright = (-9.25, 51.9)
    # w_section = cube_slice(w_cube, xs_bottomleft, xs_topright, height=(0, max_height), force_latitude=True)
    # theta_section = cube_slice(theta_cube, xs_bottomleft, xs_topright, height=(0, max_height), force_latitude=True)
    # RH_section = cube_slice(RH_cube, xs_bottomleft, xs_topright, height=(0, max_height), force_latitude=True)
    # orog_section = cube_slice(orog_cube, xs_bottomleft, xs_topright, force_latitude=True)
    # plot_xsect_latitude(w_section, theta_section, RH_section, orog_section)
