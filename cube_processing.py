import time

import cartopy.crs as ccrs
import iris
import numpy as np
from scipy.interpolate import griddata

from miscellaneous import make_great_circle_points
from plot_profile_from_UKV import convert_to_ukv_coords
from plot_xsect import get_coord_index


def cube_at_single_level(cube, map_height, bottomleft=None, topright=None):
    """
    returns the cube at a selected level_height and between bottom left and top right bounds
    Parameters
    ----------
    cube
    map_height
    bottomleft : tuple
        lon/lat for the bottom left point of the map
    topright : tuple
        lon/lat for the top right point of the map
    Returns
    -------

    """
    crs_latlon = ccrs.PlateCarree()
    crs_rotated = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    bl_model = crs_rotated.transform_point(bottomleft[0], bottomleft[1], crs_latlon)
    tr_model = crs_rotated.transform_point(topright[0], topright[1], crs_latlon)
    # TODO change level_height to altitude
    height_index = cube.coord('level_height').nearest_neighbour_index(map_height)
    single_level = cube[height_index].intersection(grid_latitude=(bl_model[1], tr_model[1]),
                                                   grid_longitude=(bl_model[0], tr_model[0]))
    return single_level

def cube_slice(cube, bottom_left, top_right, height=None, force_latitude=False):
    """
    Returns a slice of cube between bottom_left (lon, lat) and top_right corners, and between heights
    Parameters
    ----------
    force_latitude : bool
        if True will set the top_right latitude index to the bottom_left latitude index
    cube : Cube
        the cube to be sliced
    bottom_left : tuple
        the bottom left corner
    top_right : tuple
    height : tuple

    Returns
    -------

    """
    crs_latlon = ccrs.PlateCarree()
    crs_rotated = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    bl_model = convert_to_ukv_coords(*bottom_left, crs_latlon, crs_rotated)
    tr_model = convert_to_ukv_coords(*top_right, crs_latlon, crs_rotated)
    # bl_model = crs_rotated.transform_point(*bottom_left, crs_latlon)
    # tr_model = crs_rotated.transform_point(*top_right, crs_latlon)

    lat_idxs = [cube.coord('grid_latitude').nearest_neighbour_index(bl_model[1]),
                cube.coord('grid_latitude').nearest_neighbour_index(tr_model[1])]

    lon_idxs = (cube.coord('grid_longitude').nearest_neighbour_index(bl_model[0]),
                cube.coord('grid_longitude').nearest_neighbour_index(tr_model[0]))

    # only slice the height if it is given and if there is a height coordinate
    if (cube.ndim == 3) and height is not None:
        height_idxs = (cube.coord('level_height').nearest_neighbour_index(height[0]),
                       cube.coord('level_height').nearest_neighbour_index(height[1]))

        cube = cube[height_idxs[0] : height_idxs[1] + 1]

    elif height is not None and cube.ndim != 3:
        raise Exception('you gave heights but the cube is not 3 dimensional')

    if force_latitude:
        # this slices the last two dimensions regardless of how many are in front of them
        return cube[..., lat_idxs[0], lon_idxs[0] : lon_idxs[1] + 1]
    else:
        return cube[..., lat_idxs[0] : lat_idxs[1] + 1, lon_idxs[0] : lon_idxs[1] + 1]



def check_level_heights(q, t):
    """check whether q and Temperature cubes have same level heights and adjust if necessary."""
    if q.coord('level_height').points[0] == t.coord('level_height').points[0]:
        pass
    elif q.coord('level_height').points[1] == t.coord('level_height').points[0]:
        q = q[1:]
    else:
        raise ValueError('Double check the T and q level_heights - they do not match')
    return q

def cube_from_array_and_cube(array, copy_cube, unit=None, std_name=None):
    """
    Creates a new Cube by coping copy_cube and sticking in array as cube.data
    Parameters
    ----------
    a : ndarray
        data for new array
    copy_cube : Cube
        cube to be copied
    unit : str
        optional. units for new cube. if None will use copy_cube's units
    std_name : str
        optional. standard name for new cube. if None will use copy_cube's standard name

    Returns
    -------
    Cube

    """
    new_cube = copy_cube.copy()
    new_cube.data = array
    # is deleting useful in any way?
    del array
    if unit is not None:
        new_cube.units = unit
    if std_name is not None:
        new_cube.standard_name = std_name

    return new_cube


def great_circle_xsect(cube, n=50, gc_start=None, gc_end=None):
    """
    Produces an interpolated cross-section of cube along a great circle between gc_start and gc_end
    (uses the level_heights of the cube)
    Parameters
    ----------
    cube : Cube
        the cube to be interpolated
    n : int
        the number of points along the great circle at which is interpolated
    gc_start : tuple
        lon/lat of the start of the great circle
    gc_end : tuple
        lon/lat of the end of the great circle

    Returns
    -------
    ndarray
        the interpolated cross-section

    """

    crs_latlon = ccrs.PlateCarree()
    crs_rotated = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    gc = make_great_circle_points(gc_start, gc_end, n)
    gc_model = np.array([convert_to_ukv_coords(gc[0, i], gc[1, i], crs_latlon, crs_rotated) for i in range(len(gc[0]))])
    grid = np.moveaxis(np.array(np.meshgrid(cube.coord('level_height').points,
                                            cube.coord('grid_longitude').points,
                                            cube.coord('grid_latitude').points)),
                       [0, 1, 2, 3], [-1, 2, 0, 1])
    points = grid.reshape(-1, grid.shape[-1])

    broadcast_lheights = np.broadcast_to(cube.coord('level_height').points, (n, cube.coord('level_height').points.shape[0])).T
    broadcast_gc = np.broadcast_to(gc_model, (cube.coord('level_height').points.shape[0], *gc_model.shape))

    # combine
    model_gc_with_heights = np.concatenate((broadcast_lheights[:, :, np.newaxis], broadcast_gc), axis=-1)

    start_time = time.clock()
    print('start interpolation...')
    xsect = griddata(points, cube[:, ::-1].data.flatten(), model_gc_with_heights)
    print(f'{time.clock() - start_time} seconds needed to interpolate')

    return xsect

def add_orography(orography_cube, *cubes):
    orog_coord = iris.coords.AuxCoord(orography_cube.data, standard_name=str(orography_cube.standard_name),
                                      long_name='orography', var_name='orog', units=orography_cube.units)
    for cube in cubes:
        sigma = cube.coord('sigma')
        delta = cube.coord('level_height')
        fac = iris.aux_factory.HybridHeightFactory(delta=delta, sigma=sigma, orography=orog_coord)
        cube.add_aux_coord(orog_coord, (get_coord_index(cube, 'grid_latitude'),get_coord_index(cube, 'grid_longitude')))
        cube.add_aux_factory(fac)
    del orog_coord

    return cubes