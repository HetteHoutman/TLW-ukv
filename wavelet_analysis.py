import sys

import datetime as dt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import py_cwt2d
from skimage.filters import gaussian, threshold_local

from miscellaneous import check_argv_num, k_spaced_lambda, create_bins_from_midpoints
from prepare_data import get_radsim_img, get_w_field_img
from wavelet import *
from wavelet_plot import *

if __name__ == '__main__':
    # options
    test = False
    stripe_test = False
    use_radsim = False

    lambda_min = 3
    lambda_max = 35
    theta_bin_width = 5
    omega_0x = 6
    if use_radsim:
        pspec_threshold = 1e-2 # wfield thresholded
    else:
        pspec_threshold = 1e-4 # wfield unthresholded

    block_size = 51
    vertical_coord = 'air_pressure'
    analysis_level = 70000
    n_lambda = 60

    # settings
    check_argv_num(sys.argv, 3, "(datetime (YYYY-MM-DD_HH), leadtime, region)")
    datetime_string = sys.argv[1]
    datetime = dt.datetime.strptime(datetime_string, '%Y-%m-%d_%H')
    leadtime = int(sys.argv[2])
    region = sys.argv[3]

    save_path = f'./plots/{datetime_string}/{region}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if test:
        save_path = f'./plots/test/'

    if use_radsim:
        save_path += 'radsim_'

    if leadtime != 0:
        save_path += f'ld{leadtime}_'

    # produce image
    if use_radsim:
        print('Using radsim, setting leadtime=0 (currently does not support other leadtimes than 0)')
        leadtime = 0
        orig, Lx, Ly = get_radsim_img(datetime, region)
    else:
        orig, Lx, Ly = get_w_field_img(datetime, region, leadtime=leadtime, coord=vertical_coord, map_height=analysis_level)

    # normalise orig image
    orig -= orig.min()
    orig /= orig.max()

    if use_radsim:
        orig = orig > threshold_local(orig, block_size)

    lambdas, lambdas_edges = k_spaced_lambda([lambda_min, lambda_max], n_lambda)
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
            csv_file = f'radsim_henk'
        else:
            csv_file = f'ukv_normalised'

        if leadtime != 0:
            csv_file += f'_ld{leadtime}'

        csv_file += '.csv'

        try:
            df = pd.read_csv(csv_root + csv_file, index_col=[0, 1, 2], parse_dates=[0])
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'template.csv', index_col=[0, 1, 2], parse_dates=[0])

        df.sort_index(inplace=True)

        df.loc[(str(datetime.date()), region, datetime.hour), 'lambda'] = lambda_selected
        df.loc[(str(datetime.date()), region, datetime.hour), 'lambda_min'] = lambda_bounds[0]
        df.loc[(str(datetime.date()), region, datetime.hour), 'lambda_max'] = lambda_bounds[1]
        df.loc[(str(datetime.date()), region, datetime.hour), 'theta'] = theta_selected
        df.loc[(str(datetime.date()), region, datetime.hour), 'theta_min'] = theta_bounds[0]
        df.loc[(str(datetime.date()), region, datetime.hour), 'theta_max'] = theta_bounds[1]

        df.sort_index(inplace=True)
        df.to_csv(csv_root + csv_file)

        # data_root = f'/storage/silver/metstudent/phd/sw825517/ukv_pspecs/{datetime.strftime("%Y-%m-%d_%H")}_{leadtime:03d}/{region}/'
        # if not os.path.exists(data_root):
        #     os.makedirs(data_root)
        # np.save(data_root + 'pspec.npy', pspec.data)
        # np.save(data_root + 'lambdas.npy', lambdas)
        # np.save(data_root + 'thetas.npy', thetas)
        # np.save(data_root + 'threshold.npy', [pspec_threshold])
        # np.save(data_root + 'histogram.npy', strong_hist)

