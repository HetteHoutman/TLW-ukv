import matplotlib.colors as colors

def centred_cnorm(data):
    w_min = data.min()
    w_max = data.max()
    range_max = max(abs(w_max), abs(w_min))
    return colors.Normalize(vmin=-range_max, vmax=range_max)