# this file contains some useful general plotting functions
import matplotlib.colors as colors
import numpy as np


def centred_cnorm(data):
    """
    use for "norm" kwarg in plt for a colormap normalisation that is centred around zero.
    Parameters
    ----------
    data : ndarray
        the data that is plotted

    Returns
    -------
    colors.Normalize
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    range_max = max(abs(data_max), abs(data_min))
    return colors.Normalize(vmin=-range_max, vmax=range_max)