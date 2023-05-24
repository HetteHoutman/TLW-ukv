import numpy as np
from pyproj import Geod


def make_great_circle_points(start, end):
    """
    returns an array of n lon/lat pairs on great circle between (lon, lat) of start and end points
    Parameters
    ----------
    start : tuple
        (lon, lat) of start point
    end : tuple
        (lon, lat) of end point

    Returns
    -------
    ndarray
        lon/lat pairs of points on great circle
    """
    # TODO include start and end points
    g = Geod(ellps='WGS84')
    great_circle = np.array(g.npts(*start, *end, 100, initial_idx=0, terminus_idx=0)).T
    return great_circle