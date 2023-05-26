import numpy as np
from iris.analysis import trajectory
from pyproj import Geod


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

def make_custom_traj(sample_points):
    """
    Returns an iris.analysis.Trajectory instance with sample points given by gc
    Parameters
    ----------
    sample_points : ndarray
        ndarray of shape (m, 2) containing the sample points for the trajectory instance.
        should be in format lon/lat

    Returns
    -------
    iris.analysis.Trajectory
    """
    waypoints = [{'grid_longitude': sample_points[0][0], 'grid_latitude': sample_points[0][1]},
                 {'grid_longitude': sample_points[-1][0], 'grid_latitude': sample_points[-1][1]}]
    traj = trajectory.Trajectory(waypoints, sample_count=sample_points.shape[0])
    # replace trajectory points which are equally spaced in lat/lon with great circle points
    for gcpoint, d in zip(sample_points, traj.sampled_points):
        d['grid_longitude'] = gcpoint[0]
        d['grid_latitude'] = gcpoint[1]

    return traj

