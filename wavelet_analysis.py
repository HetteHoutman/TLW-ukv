import sys

import datetime as dt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import py_cwt2d
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_local

from miscellaneous import check_argv_num, k_spaced_lambda, create_bins_from_midpoints, log_spaced_lambda
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
        # pspec_threshold = 1e-2 # wfield thresholded

    pspec_threshold = 1e-2

    pixels_per_km = 1
    block_size = 51
    vertical_coord = 'air_pressure'
    analysis_level = 70000
    n_lambda = 50

    # settings
    print('Running ' + sys.argv[1] + ' ' + sys.argv[2])
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

    if use_radsim:
        orig = orig > threshold_local(orig, block_size)

    factor = (lambda_max / lambda_min) ** (1 / (n_lambda -1))
    # have two spots before and after lambda range for finding local maxima
    lambdas, lambdas_edges = log_spaced_lambda([lambda_min / factor ** 2, lambda_max * factor ** 2], factor)
    thetas = np.arange(0, 180, theta_bin_width)
    thetas_edges = create_bins_from_midpoints(thetas)
    scales = omega_0x * lambdas / (2 * np.pi)

    # initialise wavelet power spectrum array and fill
    pspec = np.zeros((*orig.shape, len(lambdas), len(thetas)))
    for i, theta in enumerate(thetas):
        cwt, wavnorm = py_cwt2d.cwt_2d(orig, scales, 'morlet', omega_0x=omega_0x, phi=np.deg2rad(90 + theta), epsilon=1)
        pspec[..., i] = (abs(cwt) / scales) ** 2

    pspec /= orig.var()
    # calculate derived things
    pspec = np.ma.masked_less(pspec, pspec_threshold)

    # e-folding distance for Morlet
    efold_dist = np.sqrt(2) * scales
    coi_mask = cone_of_influence_mask(pspec.data, efold_dist, pixels_per_km)
    pspec = np.ma.masked_where(pspec.mask | coi_mask, pspec.data)

    threshold_mask_idx = np.argwhere(~pspec.mask)
    strong_lambdas, strong_thetas = lambdas[threshold_mask_idx[:, -2]], thetas[threshold_mask_idx[:, -1]]

    max_pspec = pspec.max((-2, -1))
    max_lambdas, max_thetas = max_lambda_theta(pspec, lambdas, thetas)

    avg_pspec = np.ma.masked_less(pspec.data, pspec_threshold / 2).mean((0, 1))

    # histograms
    strong_hist, _, _ = np.histogram2d(strong_lambdas, strong_thetas, bins=[lambdas_edges, thetas_edges])
    max_hist, _, _ = np.histogram2d(max_lambdas[~max_lambdas.mask].flatten(), max_thetas[~max_lambdas.mask].flatten(), bins=[lambdas_edges, thetas_edges])

    # histogram smoothing (tile along theta-axis and select middle part so that smoothing is periodic over theta)
    # TODO make sure smoothing is adaptive to resolution in lambda, theta
    strong_hist_smoothed = gaussian(np.tile(strong_hist, 3))[:, strong_hist.shape[1]:strong_hist.shape[1] * 2]
    max_hist_smoothed = gaussian(np.tile(max_hist, 3))[:, max_hist.shape[1]:max_hist.shape[1] * 2]

    # find peaks
    peak_idxs = peak_local_max(max_hist_smoothed)

    # only keep peaks which correspond to an area of larger than area_threshold times lambda^2
    area_threshold = 1
    area_condition = (max_hist_smoothed / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2 )[tuple(peak_idxs.T)] > area_threshold
    # only keep peaks within lambda range
    lambda_condition = (lambdas[peak_idxs[:,0]] >= 3) & (lambdas[peak_idxs[:,0]] <= 35)

    peak_idxs = peak_idxs[area_condition & lambda_condition]
    lambdas_selected, thetas_selected = lambdas[peak_idxs[:,0]], thetas[peak_idxs[:,1]]
    areas_selected = max_hist_smoothed[tuple(peak_idxs.T)]

    # plot images
    plot_contour_over_image(orig, max_pspec, Lx, Ly, cbarlabels=[r'Vertical velocity $\mathregular{(ms^{-1})}$', r'$\max$ $P(\lambda, \vartheta)$'],
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_pspec_max.png', dpi=300)
    plt.close()

    plot_contour_over_image(orig, max_lambdas, Lx, Ly, cbarlabels=[r'Vertical velocity $\mathregular{(ms^{-1})}$', 'Dominant wavelength (km)'],
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_lambda.png', dpi=300)
    plt.close()

    plot_contour_over_image(orig, max_thetas, Lx, Ly, cbarlabels=[r'Vertical velocity $\mathregular{(ms^{-1})}$', 'Dominant orientation (degrees from North)'],
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_theta.png', dpi=300)
    plt.close()

    # plot histograms
    plot_polar_pcolormesh(np.ma.masked_equal(strong_hist, 0) / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2, lambdas_edges, thetas_edges, cbarlabel=r'Area / $\lambda^2$', vmin=0)
    for l, t in zip(lambdas_selected, thetas_selected):
        plt.scatter(np.deg2rad(t), l, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_strong_pspec_polar.png', dpi=300)
    plt.close()

    plot_polar_pcolormesh(np.ma.masked_equal(max_hist, 0) / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2, lambdas_edges, thetas_edges, cbarlabel=r'Area / $\lambda^2$', vmin=0)
    for l, t in zip(lambdas_selected, thetas_selected):
        plt.scatter(np.deg2rad(t), l, marker='x', color='k')
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
            csv_file = f'ukv_newalg'

        if leadtime != 0:
            csv_file += f'_ld{leadtime}'

        csv_file += '.csv'

        try:
            df = pd.read_csv(csv_root + csv_file, parse_dates=[0])
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'new_template.csv', parse_dates=[0])

        # store peaks
        for l, t, area in zip(lambdas_selected, thetas_selected, areas_selected):
            df.loc[len(df)] = [datetime.date(), region, datetime.hour, l, t, area]

        df.to_csv(csv_root + csv_file, index=False)

        # data_root = f'/storage/silver/metstudent/phd/sw825517/ukv_pspecs/{datetime.strftime("%Y-%m-%d_%H")}_{leadtime:03d}/{region}/'
        # if not os.path.exists(data_root):
        #     os.makedirs(data_root)
        # np.save(data_root + 'pspec.npy', pspec.data)
        # np.save(data_root + 'lambdas.npy', lambdas)
        # np.save(data_root + 'thetas.npy', thetas)
        # np.save(data_root + 'threshold.npy', [pspec_threshold])
        # np.save(data_root + 'histogram.npy', strong_hist)

