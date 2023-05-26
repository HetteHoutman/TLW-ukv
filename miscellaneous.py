import numpy as np
from pyproj import Geod
from iris.analysis import trajectory

from plot_profile_from_UKV import convert_to_ukv_coords


def make_great_circle_points(start, end, n):
    """
    returns an array of n lon/lat pairs on great circle between (lon, lat) of start and end points
    Parameters
    ----------
    start : tuple
        (lon, lat) of start point
    end : tuple
        (lon, lat) of end point
    n : int
        number of points

    Returns
    -------
    ndarray
        lon/lat pairs of points on great circle
    """
    g = Geod(ellps='WGS84')
    great_circle = np.array(g.npts(*start, *end, n, initial_idx=0, terminus_idx=0)).T
    return great_circle

def make_great_circle_traj(gc, n):
    waypoints = [{'grid_longitude': gc[0][0], 'grid_latitude': gc[0][1]},
                 {'grid_longitude': gc[-1][0], 'grid_latitude': gc[-1][1]}]
    traj = trajectory.Trajectory(waypoints, sample_count=n)
    # replace trajectory points which are equally spaced in lat/lon with great circle points
    for gcpoint, d in zip(gc, traj.sampled_points):
        d['grid_longitude'] = gcpoint[0]
        d['grid_latitude'] = gcpoint[1]

    return traj

