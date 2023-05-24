import cartopy.crs as ccrs
import iris.coords
import iris.plot as iplt
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from pyproj import Geod

import thermodynamics as th
from cube_processing import cube_at_single_level, check_level_heights, cube_slice
from general_plotting_fns import centred_cnorm
from iris_read import *
from miscellaneous import make_great_circle_points
from plot_profile_from_UKV import convert_to_ukv_coords
from plot_xsect import get_grid_latlon_from_rotated, add_grid_latlon_to_cube, get_coord_index
from pp_processing import data_from_pp_filename
from sonde_locs import sonde_locs
from thermodynamics import potential_temperature


def plot_xsect(w_section, theta_section, RH_section, orog_section, max_height=5000, cmap=mpl_cm.get_cmap("brewer_PuOr_11"),
               coords=None):
    """plots the cross-section with filled contours of w and normal contours of theta and RH"""
    # TODO mention that this function is just plots along a latitude as a sanity check for the interpolation
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

    plt.ylim((0,max_height))
    plt.clabel(theta_con)
    plt.xlabel(f'{w_section.coord(coords[0]).name().capitalize()} / deg')
    plt.ylabel(f'{w_section.coord(coords[1]).name().capitalize()} / {str(w_section.coord(coords[1]).units)}')
    plt.title(f'Cross-section approximately along lat {lat} deg')
    plt.colorbar(w_con, label='Upward air velocity / m/s')

    plt.tight_layout()
    plt.savefig(f'plots/xsect_lat{lat}_{year}{month}{day}_{h}.png', dpi=300)
    plt.show()


def plot_xsect_map(cube_single_level, cmap=mpl_cm.get_cmap("brewer_PuOr_11"), start=(-10.35, 51.9), end=(-6, 55)):
    """
    Plots the map indicating the cross-section, in addition to the w field.
    Parameters
    ----------
    cube_single_level : Cube
        the single level cube to be plotted
    cmap :
        colors
    end : tuple
        lon/lat of the end of the great circle cross-section line to be plotted, or None if no line is to be plotted
    start : tuple
        lon/lat of the start of the great circle cross-section line to be plotted, or None if no line is to be plotted
    """

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_latlon})
    ax.coastlines()

    w_con = iplt.contourf(cube_single_level, coords=['longitude', 'latitude'],
                          cmap=cmap, norm=centred_cnorm(cube_single_level.data))
    # ax.plot(grid_latlon['true_lons'][lat_index, lon_index_west:lon_index_east + 1],
    #         grid_latlon['true_lats'][lat_index, lon_index_west:lon_index_east + 1],
    #         color='k', zorder=50)

    if (start is None) or (end is None):
        pass
    else:
        great_circle = make_great_circle_points(end, start)
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
    plt.savefig(f'plots/xsect_map_lat{lat}_{year}{month}{day}_{h}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # read file
    # TODO use os.sep to make system transferable?
    indir = '/home/users/sw825517/Documents/ukv_data/'
    filename = indir + 'prodm_op_ukv_20150414_09_004.pp'

    year, month, day, forecast_time = data_from_pp_filename(filename)
    h = 12

    w_cube = read_variable(filename, 150, h)
    t_cube = read_variable(filename, 16004, h)
    p_cube = read_variable(filename, 408, h)
    q_cube = read_variable(filename, 10, h)

    q_cube = check_level_heights(q_cube, t_cube)

    orog_file = indir + 'prods_op_ukv_20150414_09_000.pp'
    # TODO might be easier not to use read_variable but just iris.load
    orog_cube = read_variable(orog_file, 33, 9)

    # add true lat lon
    grid_latlon = get_grid_latlon_from_rotated(w_cube)
    add_grid_latlon_to_cube(w_cube, grid_latlon)
    add_grid_latlon_to_cube(p_cube, grid_latlon)
    add_grid_latlon_to_cube(orog_cube, grid_latlon)

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
    lonbound_west = -10.4
    lonbound_east = -9.4

    crs_latlon = ccrs.PlateCarree()
    crs_rotated = w_cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    model_westbound, model_lat = convert_to_ukv_coords(lonbound_west, lat, crs_latlon, crs_rotated)
    model_eastbound, temp = convert_to_ukv_coords(lonbound_east, lat, crs_latlon, crs_rotated)
    lat_index = w_cube.coord('grid_latitude').nearest_neighbour_index(model_lat)
    lon_index_west = w_cube.coord('grid_longitude').nearest_neighbour_index(model_westbound)
    lon_index_east = w_cube.coord('grid_longitude').nearest_neighbour_index(model_eastbound)

    map_height = 750
    map_bottomleft = (-10.5, 51.6)
    map_topright = (-9.25, 52.1)

    xs_bottomleft = (-10.4, 51.9)
    xs_topright = (-9.25, 51.9)
    max_height = 5000

    gc_start = (-10.4, 51.9)
    gc_end = (-9.4, 51.9)

    w_single_level = cube_at_single_level(w_cube, map_height, bottomleft=map_bottomleft, topright=map_topright)

    w_section = cube_slice(w_cube, xs_bottomleft, xs_topright, height=(0, max_height), force_latitude=True)
    theta_section = cube_slice(theta_cube, xs_bottomleft, xs_topright, height=(0, max_height), force_latitude=True)
    RH_section = cube_slice(RH_cube, xs_bottomleft, xs_topright, height=(0, max_height), force_latitude=True)
    orog_section = cube_slice(orog_cube, xs_bottomleft, xs_topright, force_latitude=True)


    plot_xsect_map(w_single_level, start=gc_start, end=gc_end)
    plot_xsect(w_section, theta_section, RH_section, orog_section)


    # ---------------- new -------------------

    w = w_cube
    n = 50

    g = Geod(ellps='WGS84')
    # TODO need to include start and end points
    gc = np.array(g.npts(*gc_start, *gc_end, n)).T
    import time
    import scipy

    start_time = time.clock()
    gc_model = np.array([crs_rotated.transform_point(gc[0,i], gc[1,i], crs_latlon) for i in range(len(gc[0]))])
    # gc_model[:,0] += 360

    bl_model = crs_rotated.transform_point(map_bottomleft[0], map_bottomleft[1], crs_latlon)
    tr_model = crs_rotated.transform_point(map_topright[0], map_topright[1], crs_latlon)

    max_height = 5000
    max_height_index = w.coord('level_height').nearest_neighbour_index(max_height * 1.1)

    w = w[:max_height_index].intersection(grid_latitude=(bl_model[1], tr_model[1]),
                                          grid_longitude=(bl_model[0], tr_model[0]))

    # grid = np.moveaxis(np.array(np.meshgrid(w.coord('grid_longitude').points, w.coord('grid_latitude').points)), 0, -1)
    grid = np.moveaxis(np.array(np.meshgrid(w.coord('level_height').points[:max_height_index],
                                            w.coord('grid_longitude').points,
                                            w.coord('grid_latitude').points)),
                       [0,1,2,3], [-1,2,0,1])

    points = grid.reshape(-1, grid.shape[-1])

    broadcast_lheights = np.broadcast_to(w.coord('level_height').points[:max_height_index], (n, max_height_index)).T
    broadcast_gc = np.broadcast_to(gc_model, (max_height_index, *gc_model.shape))
    # combine
    model_gc_with_heights = np.concatenate((broadcast_lheights[:,:,np.newaxis], broadcast_gc), axis=-1)

    start_time = time.clock()
    print('start interpolation...')
    w_slice = scipy.interpolate.griddata(points, w[:, ::-1].data.flatten(), model_gc_with_heights)
    print(f'{time.clock()-start_time} seconds needed to interpolate')
    # w_slice = np.empty((max_height_index + 1, n))
    # start_time = time.clock()
    # for k in range(max_height_index + 1):
    #     w_slice[k] = scipy.interpolate.griddata(points, w[k].data[::-1].flatten(), gc_model)
    #     if k % 5 == 0:
    #         print(f'at index {k}/{max_height_index}...')

    # print(f'{time.clock()-start_time} seconds needed to iterate over height and fill w_slice')
    print('OKAY SO SOMETHING is up with how w cube is flattened wrt to points and model_gc_with_heights'
          'need to figure out how to flatten w cube properly.')

    print('Actually: scipy.griddata might be more useful. can just give it altitude instead of level_height'
          'and will interpolate automatically. might be slow initially but can refine later?')

    print(f'ask Sue for her oblique xsect code to compare')
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