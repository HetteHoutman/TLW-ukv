import numpy as np
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