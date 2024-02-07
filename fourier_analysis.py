import sys

import cartopy.crs as ccrs
import iris
import iris.cube
import iris.plot as iplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from astropy.convolution import convolve, Gaussian2DKernel
from iris.analysis.cartography import rotate_winds
from matplotlib import colors

from cube_processing import read_variable, cube_at_single_level, create_km_cube
from fourier import *
from fourier_plot import plot_pspec_polar, plot_radial_pspec, plot_2D_pspec, filtered_inv_plot, plot_corr
from miscellaneous import check_argv_num, load_settings, get_datetime_from_settings, get_sat_map_bltr, \
    make_title_and_save_path
from prepare_data import get_radsim_img


def plot_wind(w_cube, u_cube, v_cube, empty, step=25, title='title'):
    u_rot, v_rot = rotate_winds(u_cube, v_cube, empty.coord_system())
    u_rot = u_rot.regrid(empty, iris.analysis.Linear())
    v_rot = v_rot.regrid(empty, iris.analysis.Linear())

    fig, ax = plt.subplots(1, 1,
                           subplot_kw={'projection': ccrs.PlateCarree()}
                           )
    iplt.pcolormesh(w_cube[0], norm=mpl.colors.CenteredNorm())
    iplt.quiver(u_rot[0, ::step, ::step], v_rot[0, ::step, ::step], pivot='middle')
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
    plt.savefig(save_path + 'wind_plot.png', dpi=300)
    plt.close()




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

    # plot wind comparison
    plot_wind(w_field, u_single_level, v_single_level, empty, title=my_title)

    # prepare data for fourier analysis
    Lx, Ly = extract_distances(w_field.coords('latitude')[0].points, w_field.coords('longitude')[0].points)
    w_field = w_field[0, ::-1].data
    return w_field, Lx, Ly


if __name__ == '__main__':
    # TODO look for pp files automatically based on json file dates

    # options
    # TODO check k3 k2 settings, possibly just turn into a setting which takes a power of k as an argument?
    k3 = True
    smoothed = True
    mag_filter = False
    test = False
    use_sim_sat = True

    min_lambda = 4
    max_lambda = 30
    wnum_bin_width = 0.05
    theta_bin_width = 5

    # settings
    # TODO check map and sat bl and tr, ireland e.g. is not properly adjusted - problems with interpolation onto latlon
    check_argv_num(sys.argv, 2, "(settings, region json files)")
    s = load_settings(sys.argv[1])
    datetime = get_datetime_from_settings(s)
    region = sys.argv[2]
    sat_bl, sat_tr, map_bl, map_tr = get_sat_map_bltr(region)
    my_title, save_path = make_title_and_save_path(datetime, region, 'ukv', test, smoothed, mag_filter,
                                                   k3=k3, use_sim_sat=use_sim_sat)

    # load data
    empty = create_km_cube(sat_bl, sat_tr)

    # use ukv w-field or radsim
    if not use_sim_sat:
        orig, Lx, Ly = get_w_field_img(s)
    else:
        orig, Lx, Ly = get_radsim_img(s, datetime, empty)

    # get reciprocal space coordinates
    K, L, wavenumbers, thetas = recip_space(Lx, Ly, orig.shape)
    wavelengths = 2 * np.pi / wavenumbers

    # orig = stripey_test(orig, Lx, Ly, [10, 5], [15, 135])

    # do actual fourier transform
    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    # bandpass through expected TLW wavelengths
    bandpassed = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)

    # plot ingoing data
    plt.figure()
    filtered_inv_plot(orig, bandpassed, Lx, Ly, inverse_fft=False, title=my_title, radsim=use_sim_sat # latlon=area_extent
                      )
    plt.savefig(save_path + 'sat_plot.png', dpi=300)

    # get 2d power spectrum
    pspec_2d = np.ma.masked_where(bandpassed.mask, abs(shifted_ft) ** 2)

    # multiply by |k|^2
    if k3:
        pspec_2d = np.ma.masked_where(pspec_2d.mask, pspec_2d.data * wavenumbers ** 3)

    # convert power spectrum to polar coordinates
    # noinspection PyTupleAssignmentBalance
    radial_pspec, wnum_bins, wnum_vals, theta_bins, theta_vals = make_polar_pspec(pspec_2d, wavenumbers, wnum_bin_width,
                                                                                  thetas, theta_bin_width)

    if smoothed:
        pspec_2d = np.ma.masked_where(pspec_2d.mask, convolve(pspec_2d.data, Gaussian2DKernel(7, x_size=15, y_size=15),
                                                              boundary='wrap'))
        radial_pspec = convolve(radial_pspec, Gaussian2DKernel(3, x_size=11, y_size=11), boundary='wrap')

    # find maximum in polar power spectrum
    bounded_polar_pspec, bounded_wnum_vals = apply_wnum_bounds(radial_pspec, wnum_vals, wnum_bins,
                                                               (min_lambda, max_lambda))
    # dominant_wnum_max, dominant_theta_max = find_max(bounded_polar_pspec, bounded_wnum_vals, theta_vals)
    dom_wlen_max, dom_theta_max, dom_K_max, dom_L_max = find_cart_max(pspec_2d.data, K, L, wavelengths, thetas)

    # plot polar power spectrum along with maximum
    plt.figure()
    plot_pspec_polar(wnum_bins, theta_bins, radial_pspec, scale='log', xlim=(0.05, 4.5),
                     vmin=np.nanmin(bounded_polar_pspec), vmax=np.nanmax(bounded_polar_pspec),
                     title=my_title, min_lambda=min_lambda, max_lambda=max_lambda)
    # plt.scatter(dominant_wnum_max, dominant_theta_max, marker='x', color='k', s=100, zorder=100)
    plt.tight_layout()
    plt.savefig(save_path + 'polar_pspec.png', dpi=300)

    print(f'Dominant wavelength: {dom_wlen_max:.2f} km')
    print(f'Dominant angle: {dom_theta_max:.0f} deg from north')

    # plot radial power spectrum
    plt.figure()
    plot_radial_pspec(radial_pspec, wnum_vals, theta_bins, 2*np.pi/dom_wlen_max, title=my_title)
    plt.savefig(save_path + 'radial_pspec.png', dpi=300)

    # perform correlation with ellipse
    collapsed_corr = get_ellipse_correlation(pspec_2d, thetas, (2, 25))

    # find maximum in correlation array
    dominant_wlen, dominant_theta, dom_K, dom_L = find_corr_max(collapsed_corr, K, L, wavelengths, thetas)
    dominant_wnum = 2 * np.pi / dominant_wlen
    (lambda_min, lambda_plus), (theta_min, theta_plus) = find_corr_error(collapsed_corr, K, L,
                                                                         2 * np.pi / dominant_wlen, dominant_theta)

    err_wnums = np.linspace(2*np.pi/lambda_min, 2*np.pi/lambda_plus, 500)
    err_thetas = np.linspace(theta_min, theta_plus, 500)

    # plot correlation array with maximum
    plt.figure()
    plot_corr(collapsed_corr, K, L)
    for i in range(len(err_thetas)):
        plt.scatter(*pol2cart(dominant_wnum, np.deg2rad(-(err_thetas[i] % 180 - 90))), c='k', s=0.5)
        plt.scatter(*pol2cart(err_wnums[i], np.deg2rad((-dominant_theta+90))), c='k', s=0.5)
    plt.scatter(dom_K, dom_L, marker='x')
    plt.savefig(save_path + 'corr.png', dpi=300)

    # plot cartesian power spectrum with maximum from ellipse correlation
    plt.figure()
    plot_2D_pspec(pspec_2d, K, L, wavelengths, wavelength_contours=[5, 10, 35], title=my_title)
    plt.scatter(dom_K, dom_L, marker='x', label='ellipse')
    plt.scatter(dom_K_max, dom_L_max, marker='x', c='k', label='maximum')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.savefig(save_path + '2d_pspec_withcross.png', dpi=300)

    print(f'Dominant wavelength by ellipse method: {dominant_wlen:.2f} km')
    print(f'Dominant wavelength range estimate: {lambda_min:.2f}-{lambda_plus:.2f} km')
    print(f'Dominant angle by ellipse method: {dominant_theta:.0f} deg from north')
    print(f'Dominant angle range estimate: {theta_min:.0f}-{theta_plus:.0f} deg')

    # save to csv with results
    if not test:
        csv_root = 'fourier_results/'
        csv_file = 'ukv'
        if use_sim_sat:
            csv_file += '_radsim'
        if k3:
            csv_file += '_k3'
        if smoothed:
            csv_file += '_smoothed'

        csv_file += '_results.csv'

        try:
            df = pd.read_csv(csv_root + csv_file, index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'template.csv', index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)

        df.sort_index(inplace=True)
        date = pd.to_datetime(f'{s.year}-{s.month:02d}-{s.day:02d}')

        df.loc[(date, region, s.h), 'lambda'] = dominant_wlen
        df.loc[(date, region, s.h), 'lambda_min'] = lambda_min
        df.loc[(date, region, s.h), 'lambda_max'] = lambda_plus
        df.loc[(date, region, s.h), 'theta'] = dominant_theta
        df.loc[(date, region, s.h), 'theta_min'] = theta_min
        df.loc[(date, region, s.h), 'theta_max'] = theta_plus

        # sort for clarity if any new dates have been added
        df.sort_index(inplace=True)
        df.to_csv(csv_root + csv_file)

    np.save(save_path + 'pspec_array.npy', pspec_2d.data)

# TODO put plotting stuff in another file? so that that won't slow down running of file
