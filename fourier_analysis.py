import sys

import cartopy.crs as ccrs
import iris.cube
import iris.plot as iplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
from astropy.convolution import convolve, Gaussian2DKernel
from iris.analysis.cartography import rotate_winds
from matplotlib import colors

from cube_processing import read_variable, cube_at_single_level, add_orography, add_true_latlon_coords, \
    create_latlon_cube
from fourier import *
from fourier_plot import plot_pspec_polar, plot_radial_pspec, plot_2D_pspec, filtered_inv_plot, plot_corr
from miscellaneous import check_argv_num, load_settings, get_region_var


def plot_wind(w_cube, u_cube, v_cube, step=25, title='title'):
    fig, ax = plt.subplots(1, 1,
                           subplot_kw={'projection': ccrs.PlateCarree()}
                           )
    iplt.pcolormesh(w_cube[0], norm=mpl.colors.CenteredNorm())
    iplt.quiver(u_cube[0, ::step, ::step], v_cube[0, ::step, ::step], pivot='middle')
    ax.gridlines(draw_labels=True)
    ax.coastlines()
    ax.set_xlabel('True Longitude / deg')
    ax.set_ylabel('True Latitude / deg')
    plt.colorbar(label='Upward air velocity / m/s',
                 location='bottom',
                 # orientation='vertical'
                 )
    # plt.xlim(lon_bounds)
    # plt.ylim(lat_bounds)
    plt.title(title)


def get_datetime_from_settings(settings):
    return f'{settings.year}-{settings.month:02d}-{settings.day:02d}_{settings.h}'


def get_sat_map_bltr(region, region_root='/home/users/sw825517/Documents/tephiplot/regions/'):
    sat_bounds = get_region_var("sat_bounds", region, region_root)
    satellite_bottomleft, satellite_topright = sat_bounds[:2], sat_bounds[2:]
    map_bounds = get_region_var("map_bounds", region, region_root)
    map_bottomleft, map_topright = map_bounds[:2], map_bounds[2:]

    return satellite_bottomleft, satellite_topright, map_bottomleft, map_topright


def make_title_and_save_path(datetime, region, data_source_string, test, k2, smoothed, mag_filter):
    my_title = f'{datetime}_{region}_{data_source_string}'

    save_path = f'plots/{datetime}/{region}/'

    if not os.path.exists('plots/' + datetime):
        os.makedirs('plots/' + datetime)

    if test:
        save_path = f'plots/test/'
        my_title += '_test'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if k2:
        save_path += 'k2_'
        my_title += '_k2'
    if smoothed:
        save_path += 'smoothed_'
        my_title += '_smoothed'
    if mag_filter:
        save_path += 'magfiltered_'
        my_title += '_magfiltered'

    return my_title, save_path


if __name__ == '__main__':
    # TODO clean and put in functions
    # TODO look for pp files automatically based on json file dates

    # options
    k2 = False
    smoothed = True
    mag_filter = False
    test = True

    # settings
    check_argv_num(sys.argv, 2, "(settings, region json files)")
    s = load_settings(sys.argv[1])
    datetime = get_datetime_from_settings(s)
    region = sys.argv[2]

    sat_bl, sat_tr, map_bl, map_tr = get_sat_map_bltr(region)

    my_title, save_path = make_title_and_save_path(datetime, region, 'ukv', test, k2, smoothed, mag_filter)

    # load data
    w_cube = read_variable(s.file, 150, s.h)
    u_cube = read_variable(s.file, 2, s.h).regrid(w_cube, iris.analysis.Linear())
    v_cube = read_variable(s.file, 3, s.h).regrid(w_cube, iris.analysis.Linear())

    w_single_level, u_single_level, v_single_level = cube_at_single_level(s.map_height, w_cube, u_cube, v_cube,
                                                                          bottomleft=map_bl, topright=map_tr)

    empty = create_latlon_cube(sat_bl, sat_tr, n=501)
    orig = w_single_level.regrid(empty, iris.analysis.Linear())

    # plot wind comparison
    u_rot, v_rot = rotate_winds(u_single_level, v_single_level, empty.coord_system())
    u_rot = u_rot.regrid(empty, iris.analysis.Linear())
    v_rot = v_rot.regrid(empty, iris.analysis.Linear())
    plot_wind(orig, u_rot, v_rot, title=my_title)
    plt.savefig(save_path + 'wind_plot.png', dpi=300)

    # prepare data for fourier analysis
    Lx, Ly = extract_distances(orig.coords('latitude')[0].points, orig.coords('longitude')[0].points)
    orig = orig[0, ::-1].data

    # get reciprocal space coordinates
    K, L, wavenumbers, thetas = recip_space(Lx, Ly, orig.shape)
    wavelengths = 2 * np.pi / wavenumbers

    # orig = stripey_test(orig, Lx, Ly, [10, 5], [15, 135])

    # do actual fourier transform
    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    # bandpass through expected TLW wavelengths
    min_lambda = 4
    max_lambda = 30
    bandpassed = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)

    # plot ingoing data
    plt.figure()
    filtered_inv_plot(orig, bandpassed, Lx, Ly, inverse_fft=False, title=my_title,  # latlon=area_extent
                      )
    plt.savefig(save_path + 'sat_plot.png', dpi=300)

    # get 2d power spectrum
    pspec_2d = np.ma.masked_where(bandpassed.mask, abs(shifted_ft) ** 2)

    # multiply by |k|^2
    if k2:
        pspec_2d = np.ma.masked_where(pspec_2d.mask, pspec_2d.data * wavenumbers ** 2)

    # convert power spectrum to polar coordinates
    wnum_bin_width = 0.05
    theta_bin_width = 5
    # noinspection PyTupleAssignmentBalance
    radial_pspec, wnum_bins, wnum_vals, theta_bins, theta_vals = make_polar_pspec(pspec_2d, wavenumbers, wnum_bin_width,
                                                                                  thetas, theta_bin_width)

    if smoothed:
        pspec_2d = np.ma.masked_where(pspec_2d.mask, convolve(pspec_2d.data, Gaussian2DKernel(7, x_size=15, y_size=15),
                                                              boundary='wrap'))
        radial_pspec = convolve(radial_pspec, Gaussian2DKernel(3, x_size=11, y_size=11), boundary='wrap')

    # plot (smoothed) cartesian power spectrum
    plt.figure()
    plot_2D_pspec(pspec_2d, Lx, Ly, wavelength_contours=[5, 10, 35], title=my_title)
    plt.savefig(save_path + '2d_pspec.png', dpi=300)

    # find maximum in polar power spectrum
    bounded_polar_pspec, bounded_wnum_vals = apply_wnum_bounds(radial_pspec, wnum_vals, wnum_bins,
                                                               (min_lambda, max_lambda))
    dominant_wnum, dominant_theta = find_max(bounded_polar_pspec, bounded_wnum_vals, theta_vals)

    # plot polar power spectrum along with maximum
    plt.figure()
    plot_pspec_polar(wnum_bins, theta_bins, radial_pspec, scale='log', xlim=(0.05, 4.5),
                     vmin=np.nanmin(bounded_polar_pspec), vmax=np.nanmax(bounded_polar_pspec),
                     title=my_title, min_lambda=min_lambda, max_lambda=max_lambda)
    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    plt.tight_layout()
    plt.savefig(save_path + 'polar_pspec.png', dpi=300)

    print(f'Dominant wavelength: {2 * np.pi / dominant_wnum:.2f} km')
    print(f'Dominant angle: {dominant_theta:.0f} deg from north')

    # plot radial power spectrum
    plt.figure()
    plot_radial_pspec(radial_pspec, wnum_vals, theta_bins, dominant_wnum, title=my_title)
    plt.savefig(save_path + 'radial_pspec.png', dpi=300)
    # plt.show()

    # perform correlation with ellipse
    corr = correlate_ellipse(pspec_2d, thetas, (2, 25))
    rot_left_half = rotate(corr[:, :corr.shape[1] // 2 + 1], 180)
    collapsed_corr = corr[:, corr.shape[1] // 2:] + rot_left_half

    # mask bottom half of k_x = 0 line as this is the same as the top half
    collapsed_corr.mask[collapsed_corr.shape[0] // 2:, 0] = True

    # find maximum in correlation array
    idxs = np.unravel_index(collapsed_corr.argmax(), collapsed_corr.shape)
    dom_K, dom_L = K[idxs[0], K.shape[1] // 2 + idxs[1]], L[::-1][idxs[0], L.shape[1] // 2 + idxs[1]]
    dominant_wlen = wavelengths[:, wavelengths.shape[1] // 2:][idxs[0], idxs[1]]
    dominant_theta = thetas[:, thetas.shape[1] // 2:][idxs[0], idxs[1]]

    # plot correlation array with maximum
    # TODO dom scatter not right
    plt.figure()
    plot_corr(collapsed_corr, L, K)
    plt.scatter(dom_K, dom_L, marker='x')
    plt.savefig(save_path + 'corr.png', dpi=300)

    # plot 2d power spectrum with maximum from ellipse correlation
    plt.figure()
    plot_2D_pspec(pspec_2d, Lx, Ly, wavelength_contours=[5, 10, 35], title=my_title)
    plt.scatter(dom_K, dom_L, marker='x')
    plt.savefig(save_path + '2d_pspec_withcross.png', dpi=300)

    print(f'Dominant wavelength by ellipse method: {dominant_wlen:.2f} km')
    print(f'Dominant angle by ellipse method: {dominant_theta:.0f} deg from north')

    if not test:
        df = pd.read_excel('../sat_vs_ukv_results.xlsx', index_col=[0, 1])
        df.loc[(f'{s.year}-{s.month:02d}-{s.day:02d}', region), 'ukv_lambda_ellipse'] = dominant_wlen
        df.loc[(f'{s.year}-{s.month:02d}-{s.day:02d}', region), 'ukv_theta_ellipse'] = dominant_theta
        df.to_excel('../sat_vs_ukv_results.xlsx')
