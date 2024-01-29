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
from wavelet import *
from wavelet_plot import *


def get_w_field_img(settings, leadtime=0):
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

    lambda_min = 5
    lambda_max = 35
    lambda_bin_width = 1
    theta_bin_width = 5
    omega_0x = 6
    # pspec_threshold = 3e-4
    pspec_threshold = 1e-2
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

    # load data
    empty = create_km_cube(sat_bl, sat_tr)

    # produce image
    orig, Lx, Ly = get_w_field_img(s)

    # normalise orig image
    orig /= orig.max()
    # orig = exposure.equalize_hist(orig)
    orig = orig > threshold_local(orig, block_size)

    # lambdas, lambdas_edges = create_range_and_bin_edges_from_minmax([lambda_min, lambda_max],
    #                                                                 lambda_max - lambda_min + 1)
    lambdas, lambdas_edges = log_spaced_lambda([lambda_min, lambda_max], 1.075)
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

    # histogram smoothing (tile along theta-axis and select middle part so that smoothing is periodic over theta
    strong_hist_smoothed = gaussian(np.tile(strong_hist, 3))[:, strong_hist.shape[1]:strong_hist.shape[1] * 2]
    max_hist_smoothed = gaussian(np.tile(max_hist, 3))[:, max_hist.shape[1]:max_hist.shape[1] * 2]

    # determine maximum in smoothed image
    idxs = np.unravel_index(strong_hist_smoothed.argmax(), strong_hist_smoothed.shape)
    result_lambda, result_theta = lambdas[idxs[0]], thetas[idxs[1]]

    # avg max
    avg_idxs = np.unravel_index(avg_pspec.argmax(), avg_pspec.shape)
    henk_lambda, henk_theta = lambdas[avg_idxs[0]], thetas[avg_idxs[1]]

    # plot images
    plot_contour_over_image(orig, max_pspec, Lx, Ly, cbarlabel='Maximum of wavelet power spectrum',
                            alpha=0.5, norm=mpl.colors.LogNorm())
    plt.savefig(save_path + 'wavelet_pspec_max.png', dpi=300)
    plt.show()

    plot_contour_over_image(orig, max_lambdas, Lx, Ly, cbarlabel='Dominant wavelength (km)',
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_lambda.png', dpi=300)
    plt.show()

    plot_contour_over_image(orig, max_thetas, Lx, Ly, cbarlabel='Dominant orientation (degrees from North)',
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_theta.png', dpi=300)
    plt.show()

    # plot histograms
    plot_k_histogram(max_lambdas[~max_pspec.mask], max_thetas[~max_pspec.mask], lambdas_edges, thetas_edges)
    plt.savefig(save_path + 'wavelet_k_histogram_max.png', dpi=300)
    plt.show()

    plot_k_histogram(strong_lambdas, strong_thetas, lambdas_edges, thetas_edges)
    plt.scatter(result_lambda, result_theta, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_full_pspec.png', dpi=300)
    plt.show()

    plot_polar_pcolormesh(strong_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.scatter(np.deg2rad(result_theta), result_lambda, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_strong_pspec_polar.png', dpi=300)
    plt.show()

    plot_polar_pcolormesh(max_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.savefig(save_path + 'wavelet_k_histogram_max_pspec_polar.png', dpi=300)
    plt.show()

    plot_polar_pcolormesh(avg_pspec, lambdas_edges, thetas_edges, cbarlabel='Average power spectrum')
    plt.savefig(save_path + 'wavelet_average_pspec.png', dpi=300)
    plt.show()

    # save results
    if not test:
        csv_root = 'wavelet_results/'
        csv_file = f'ukv_adapt_thresh_{block_size}.csv'

        try:
            df = pd.read_csv(csv_root + csv_file, index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'template.csv', index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)

        df.sort_index(inplace=True)
        date = pd.to_datetime(f'{s.year}-{s.month:02d}-{s.day:02d}')

        df.loc[(date, region, s.h), 'lambda'] = result_lambda
        df.loc[(date, region, s.h), 'theta'] = result_theta

        df.sort_index(inplace=True)
        df.to_csv(csv_root + csv_file)

        csv_file = f'ukv_test.csv'

        try:
            df = pd.read_csv(csv_root + csv_file, index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'template.csv', index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)

        df.sort_index(inplace=True)
        date = pd.to_datetime(f'{s.year}-{s.month:02d}-{s.day:02d}')

        df.loc[(date, region, s.h), 'lambda'] = henk_lambda
        df.loc[(date, region, s.h), 'theta'] = henk_theta

        df.sort_index(inplace=True)
        df.to_csv(csv_root + csv_file)
