import numpy as np

# TODO possible split functions between different files if this file gets too long/general

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
        of shape (2, n). lon/lat pairs of points on great circle
    ndarray
        of shape (n). corresponding distances of points along the great circle from start
    """
    from pyproj import Geod
    # TODO copy that web source and change n to desired km resolution?
    g = Geod(ellps='WGS84')
    _, _, dist = g.inv(*start, *end)
    distances = np.linspace(0, dist, n)
    great_circle = np.array(g.npts(*start, *end, n, initial_idx=0, terminus_idx=0)).T
    return great_circle, distances

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
    from iris.analysis import trajectory
    waypoints = [{'grid_longitude': sample_points[0][0], 'grid_latitude': sample_points[0][1]},
                 {'grid_longitude': sample_points[-1][0], 'grid_latitude': sample_points[-1][1]}]
    traj = trajectory.Trajectory(waypoints, sample_count=sample_points.shape[0])
    # replace trajectory points which are equally spaced in lat/lon with great circle points
    for gcpoint, d in zip(sample_points, traj.sampled_points):
        d['grid_longitude'] = gcpoint[0]
        d['grid_latitude'] = gcpoint[1]

    return traj


def convert_to_ukv_coords(x, y, in_crs, out_crs):
    """transforms coordinates given in crs in_crs to coordinates in crs out_crs.
    works at least for UKV rotated pole."""
    out_x, out_y = out_crs.transform_point(x, y, in_crs)
    return out_x + 360, out_y

def convert_list_to_ukv_coords(x_list, y_list, in_crs, out_crs):
    """list version of convert_to_ukv_coords"""
    return np.array([convert_to_ukv_coords(x, y, in_crs, out_crs) for x, y in zip(x_list, y_list)])

def index_selector(desired_value, array):
    """returns the index of the value in array that is closest to desired_value"""
    return (np.abs(array - desired_value)).argmin()

