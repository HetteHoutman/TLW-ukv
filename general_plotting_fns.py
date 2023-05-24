import matplotlib.colors as colors

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
    data_min = data.min()
    data_max = data.max()
    range_max = max(abs(data_max), abs(data_min))
    return colors.Normalize(vmin=-range_max, vmax=range_max)