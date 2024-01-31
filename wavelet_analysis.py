import iris
import iris.cube
import pandas as pd
import py_cwt2d
from iris.analysis.cartography import rotate_winds
from skimage.filters import gaussian, threshold_local
import matplotlib as mpl

from cube_processing import read_variable, cube_at_single_level, create_km_cube
from fourier import *
from miscellaneous import *
from miscellaneous import check_argv_num, load_settings, get_datetime_from_settings, get_sat_map_bltr
from prepare_data import get_radsim_img
from wavelet import *
from wavelet_plot import *



def get_w_field_img(settings, leadtime=0):
    # TODO put in useful_code
    """
    gets w field from ukv and prepares it for fourier analysis
    Parameters
    ----------
    settings

    Returns
    -------

    """
    try:
        file = settings.file
    except AttributeError:
        file = f'/home/users/sw825517/Documents/ukv_data/ukv_{s.year}-{s.month:02d}-{s.day:02d}_{s.h:02d}_{leadtime:03.0f}.pp'

    w_cube = read_variable(file, 150, settings.h)
    u_cube = read_variable(file, 2, settings.h).regrid(w_cube, iris.analysis.Linear())
    v_cube = read_variable(file, 3, settings.h).regrid(w_cube, iris.analysis.Linear())
    w_single_level, u_single_level, v_single_level = cube_at_single_level(s.map_height, w_cube, u_cube, v_cube,
                                                                          bottomleft=map_bl, topright=map_tr)
    w_field = w_single_level.regrid(empty, iris.analysis.Linear())


    # prepare data for fourier analysis
    Lx, Ly = extract_distances(w_field.coords('latitude')[0].points, w_field.coords('longitude')[0].points)
    w_field = w_field[0, ::-1].data
    return w_field, Lx, Ly


if __name__ == '__main__':
    # options
    test = False
    stripe_test = False
    use_radsim = False

    lambda_min = 5
    lambda_max = 35
    lambda_bin_width = 1
    theta_bin_width = 5
    omega_0x = 6
    pspec_threshold = 1e-4 # wfield unthresholded
    # pspec_threshold = 1e-2 #wfield thresholded
    block_size = 51

    # settings
    check_argv_num(sys.argv, 2, "(settings, region json files)")
    s = load_settings(sys.argv[1])
    datetime = get_datetime_from_settings(s)
    region = sys.argv[2]
    sat_bl, sat_tr, map_bl, map_tr = get_sat_map_bltr(region)

    save_path = f'./plots/{datetime}/{region}/'
    if test:
        save_path = f'./plots/test/'

    if use_radsim:
        save_path += 'radsim'

    # load data
    empty = create_km_cube(sat_bl, sat_tr)

    # produce image
    if use_radsim:
        orig, Lx, Ly = get_radsim_img(s, datetime, empty)
    else:
        orig, Lx, Ly = get_w_field_img(s)

    # normalise orig image
    orig -= orig.min()
    orig /= orig.max()
    # orig = exposure.equalize_hist(orig)
    # orig = orig > threshold_local(orig, block_size)
    # orig = orig > th

    # lambdas, lambdas_edges = create_range_and_bin_edges_from_minmax([lambda_min, lambda_max],
    #                                                                 lambda_max - lambda_min + 1)
    # lambdas, lambdas_edges = log_spaced_lambda([lambda_min, lambda_max], 1.075)
    lambdas, lambdas_edges = k_spaced_lambda([lambda_min, lambda_max], 40)
    thetas = np.arange(0, 180, theta_bin_width)
    thetas_edges = create_bins_from_midpoints(thetas)
    scales = omega_0x * lambdas / (2 * np.pi)

    # initialise wavelet power spectrum array and fill
    pspec = np.zeros((*orig.shape, len(lambdas), len(thetas)))
    for i, theta in enumerate(thetas):
        cwt, wavnorm = py_cwt2d.cwt_2d(orig, scales, 'morlet', omega_0x=omega_0x, phi=np.deg2rad(90 + theta), epsilon=1)
        pspec[..., i] = (abs(cwt) / scales / wavnorm) ** 2

    # calculate derived things
    pspec = np.ma.masked_less(pspec, pspec_threshold)
    threshold_mask_idx = np.argwhere(~pspec.mask)
    strong_lambdas, strong_thetas = lambdas[threshold_mask_idx[:, -2]], thetas[threshold_mask_idx[:, -1]]

    max_pspec = np.ma.masked_less(pspec.data.max((-2, -1)), pspec_threshold)
    max_lambdas, max_thetas = max_lambda_theta(pspec.data, lambdas, thetas)

    avg_pspec = np.ma.masked_less(pspec.data, pspec_threshold / 2).mean((0, 1))

    # histograms
    strong_hist, _, _ = np.histogram2d(strong_lambdas, strong_thetas, bins=[lambdas_edges, thetas_edges])
    max_hist, _, _ = np.histogram2d(max_lambdas[~max_pspec.mask], max_thetas[~max_pspec.mask],
                                    bins=[lambdas_edges, thetas_edges])

    # histogram smoothing (tile along theta-axis and select middle part so that smoothing is periodic over theta)
    # TODO make sure smoothing is adaptive to resolution in lambda, theta
    strong_hist_smoothed = gaussian(np.tile(strong_hist, 3))[:, strong_hist.shape[1]:strong_hist.shape[1] * 2]
    max_hist_smoothed = gaussian(np.tile(max_hist, 3))[:, max_hist.shape[1]:max_hist.shape[1] * 2]

    # determine maximum in smoothed histogram
    lambda_selected, theta_selected, lambda_bounds, theta_bounds = find_polar_max_and_error(strong_hist_smoothed,
                                                                                            lambdas, thetas)

    # plot images
    plot_contour_over_image(orig, max_pspec, Lx, Ly, cbarlabel='Maximum of wavelet power spectrum',
                            alpha=0.5, norm=mpl.colors.LogNorm())
    plt.savefig(save_path + 'wavelet_pspec_max.png', dpi=300)
    plt.close()

    plot_contour_over_image(orig, max_lambdas, Lx, Ly, cbarlabel='Dominant wavelength (km)',
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_lambda.png', dpi=300)
    plt.close()

    plot_contour_over_image(orig, max_thetas, Lx, Ly, cbarlabel='Dominant orientation (degrees from North)',
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_theta.png', dpi=300)
    plt.close()

    # plot histograms
    plot_k_histogram(max_lambdas[~max_pspec.mask], max_thetas[~max_pspec.mask], lambdas_edges, thetas_edges)
    plt.savefig(save_path + 'wavelet_k_histogram_max.png', dpi=300)
    plt.close()

    plot_k_histogram(strong_lambdas, strong_thetas, lambdas_edges, thetas_edges)
    plt.scatter(lambda_selected, theta_selected, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_full_pspec.png', dpi=300)
    plt.close()

    plot_polar_pcolormesh(strong_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.scatter(np.deg2rad(theta_selected), lambda_selected, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_strong_pspec_polar.png', dpi=300)
    plt.close()

    plot_polar_pcolormesh(max_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.savefig(save_path + 'wavelet_k_histogram_max_pspec_polar.png', dpi=300)
    plt.close()

    plot_polar_pcolormesh(avg_pspec, lambdas_edges, thetas_edges, cbarlabel='Average power spectrum')
    plt.savefig(save_path + 'wavelet_average_pspec.png', dpi=300)
    plt.close()

    # save results
    if not test:
        csv_root = 'wavelet_results/'
        if use_radsim:
            csv_file = f'radsim_normalised.csv'
        else:
            csv_file = f'ukv_normalised.csv'

        try:
            df = pd.read_csv(csv_root + csv_file, index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'template.csv', index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)

        df.sort_index(inplace=True)
        date = pd.to_datetime(f'{s.year}-{s.month:02d}-{s.day:02d}')

        df.loc[(date, region, s.h), 'lambda'] = lambda_selected
        df.loc[(date, region, s.h), 'lambda_min'] = lambda_bounds[0]
        df.loc[(date, region, s.h), 'lambda_max'] = lambda_bounds[1]
        df.loc[(date, region, s.h), 'theta'] = theta_selected
        df.loc[(date, region, s.h), 'theta_min'] = theta_bounds[0]
        df.loc[(date, region, s.h), 'theta_max'] = theta_bounds[1]

        df.sort_index(inplace=True)
        df.to_csv(csv_root + csv_file)

