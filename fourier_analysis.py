import os
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

from cube_processing import read_variable, cube_at_single_level, create_latlon_cube, cube_from_array_and_cube
from fourier import *
from fourier_plot import plot_pspec_polar, plot_radial_pspec, plot_2D_pspec, filtered_inv_plot, plot_corr
from miscellaneous import check_argv_num, load_settings, get_datetime_from_settings, get_sat_map_bltr, \
    make_title_and_save_path
from prepare_radsim_array import get_refl
from psd import periodic_smooth_decomp
from regrid_and_save import regrid_10m_wind_and_append


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


def get_radsim_img(settings, datetime):
    """
    gets the simulated satellite imagery from radsim and prepares it for fourier analysis.
    could probs divvy this up into functions too
    Parameters
    ----------
    settings
    datetime

    Returns
    -------

    """
    # this should be the radsim output netCDF4 file
    nc_file_root = f"/home/users/sw825517/radsim/radsim-3.2/outputs"
    nc_filename = f'{datetime}.nc'

    # in case radsim has already run simulation
    try:
        refl = get_refl(nc_file_root + '/' + nc_filename)

    # if not, then run radsim
    except FileNotFoundError:
        print('Radsim output .nc file not found: running radsim to create')
        radsim_run_file = "/home/users/sw825517/radsim/radsim-3.2/src/scripts/radsim_run.py"

        # run radsim_run.py with radsim_settings, so set radsim_settings accordingly
        radsim_settings = {'config_file': f'/home/users/sw825517/radsim/radsim-3.2/outputs/{datetime}.cfg',
                           'radsim_bin_dir': '/home/users/sw825517/radsim/radsim-3.2/bin/',
                           'model_datafile': f'/home/users/sw825517/Documents/ukv_data/ukv_{datetime}_000.unpacked.pp',
                           'model_filetype': 0,
                           'rttov_coeffs_dir': '/home/users/sw825517/rttov13/rtcoef_rttov13/rttov13pred54L',
                           'rttov_coeffs_options': '_o3co2',
                           'rttov_sccld_dir': '/home/users/sw825517/rttov13/rtcoef_rttov13/cldaer_visir/',
                           'platform': "msg",
                           'satid': 3,
                           'inst': "seviri",
                           'channels': 12,
                           'output_mode': 1,
                           'addsolar': True,
                           'ir_addclouds': True,
                           'output_dir': nc_file_root,
                           'output_file': nc_filename,
                           'write_latlon': True,
                           # 'run_mfasis': True,
                           # 'rttov_mfasis_nn_dir': '/home/users/sw825517/rttov13/rtcoef_rttov13/mfasis_nn/'
                           }

        # check whether unpacked pp file exists, if not then unpack packed pp file
        if not os.path.isfile(radsim_settings['model_datafile']):
            print('Unpacked .pp file does not exist, will try to create...')
            packed_pp = f'/home/users/sw825517/Documents/ukv_data/ukv_{datetime}_000.pp'

            # check whether packed pp exists
            if os.path.isfile(packed_pp):
                # ensure packed pp has 10m winds on correct grid
                try:
                    _ = read_variable(packed_pp, 3225, settings.h)
                    _ = read_variable(packed_pp, 3226, settings.h)
                except IndexError:
                    print(f'packed .pp {packed_pp} does not have 10m winds on correct grid, regridding...')
                    regrid_10m_wind_and_append(settings, packed_pp)

                # unpack
                os.system(f'/home/users/sw825517/Documents/ukv_data/pp_unpack {packed_pp}')
                # rename unpacked pp so that it ends on '.pp'
                os.system(
                    f"cp /home/users/sw825517/Documents/ukv_data/ukv_{datetime}_000.pp.unpacked /home/users/sw825517/Documents/ukv_data/ukv_{datetime}_000.unpacked.pp")
                os.system(f"rm /home/users/sw825517/Documents/ukv_data/ukv_{datetime}_000.pp.unpacked")

            else:
                print(f'packed .pp {packed_pp} not found')
                sys.exit(1)

        set_str = ''

        for setting in radsim_settings:
            set_str += f'--{setting} {radsim_settings[setting]} '

        os.system(f"python {radsim_run_file} {set_str}")
        refl = get_refl(nc_file_root + '/' + nc_filename)

    # convert radsim reflectivity data from netCDF4 into iris cube, to regrid it onto a regular latlon grid
    surf_t = read_variable(s.file, 24, s.h)
    refl_cube = cube_from_array_and_cube(refl[::-1], surf_t, unit=1, std_name='toa_bidirectional_reflectance')
    empty = create_latlon_cube(sat_bl, sat_tr, n=501)
    refl_regrid = refl_cube.regrid(empty, iris.analysis.Linear())

    x_dist, y_dist = extract_distances(refl_regrid.coords('latitude')[0].points,
                                       refl_regrid.coords('longitude')[0].points)
    image = refl_regrid.data[::-1]
    image, smooth = periodic_smooth_decomp(image)

    return image, x_dist, y_dist


def get_w_field_img(settings):
    """
    gets w field from ukv and prepares it for fourier analysis
    Parameters
    ----------
    settings

    Returns
    -------

    """
    w_cube = read_variable(settings.file, 150, settings.h)
    u_cube = read_variable(settings.file, 2, settings.h).regrid(w_cube, iris.analysis.Linear())
    v_cube = read_variable(settings.file, 3, settings.h).regrid(w_cube, iris.analysis.Linear())
    w_single_level, u_single_level, v_single_level = cube_at_single_level(s.map_height, w_cube, u_cube, v_cube,
                                                                          bottomleft=sat_bl, topright=sat_tr)
    w_field = w_single_level.regrid(empty_latlon, iris.analysis.Linear())

    # plot wind comparison
    plot_wind(w_field, u_single_level, v_single_level, empty_latlon, title=my_title)

    # prepare data for fourier analysis
    Lx, Ly = extract_distances(w_field.coords('latitude')[0].points, w_field.coords('longitude')[0].points)
    w_field = w_field[0, ::-1].data
    return w_field, Lx, Ly


if __name__ == '__main__':
    # TODO look for pp files automatically based on json file dates

    # options
    k2 = True
    smoothed = True
    mag_filter = False
    test = True
    use_sim_sat = True

    min_lambda = 4
    max_lambda = 30
    wnum_bin_width = 0.05
    theta_bin_width = 5

    # settings
    check_argv_num(sys.argv, 2, "(settings, region json files)")
    s = load_settings(sys.argv[1])
    datetime = get_datetime_from_settings(s)
    region = sys.argv[2]
    sat_bl, sat_tr, map_bl, map_tr = get_sat_map_bltr(region)
    my_title, save_path = make_title_and_save_path(datetime, region, 'ukv', test, k2, smoothed, mag_filter,
                                                   use_sim_sat=use_sim_sat)

    # load data
    empty_latlon = create_latlon_cube(sat_bl, sat_tr, n=501)

    # use ukv w-field or radsim
    if not use_sim_sat:
        orig, Lx, Ly = get_w_field_img(s)
    else:
        orig, Lx, Ly = get_radsim_img(s, datetime)

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
    filtered_inv_plot(orig, bandpassed, Lx, Ly, inverse_fft=False, title=my_title,  # latlon=area_extent
                      )
    plt.savefig(save_path + 'sat_plot.png', dpi=300)

    # get 2d power spectrum
    pspec_2d = np.ma.masked_where(bandpassed.mask, abs(shifted_ft) ** 2)

    # multiply by |k|^2
    if k2:
        pspec_2d = np.ma.masked_where(pspec_2d.mask, pspec_2d.data * wavenumbers ** 2)

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

    # perform correlation with ellipse
    collapsed_corr = get_ellipse_correlation(pspec_2d, thetas, (2, 25))

    # find maximum in correlation array
    dominant_wlen, dominant_theta, dom_K, dom_L = find_corr_max(collapsed_corr, K, L, wavelengths, thetas)

    # plot correlation array with maximum
    plt.figure()
    plot_corr(collapsed_corr, K, L)
    plt.scatter(dom_K, dom_L, marker='x')
    plt.savefig(save_path + 'corr.png', dpi=300)

    # plot cartesian power spectrum with maximum from ellipse correlation
    plt.figure()
    plot_2D_pspec(pspec_2d, K, L, wavelengths, wavelength_contours=[5, 10, 35], title=my_title)
    plt.scatter(dom_K, dom_L, marker='x')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.savefig(save_path + '2d_pspec_withcross.png', dpi=300)

    print(f'Dominant wavelength by ellipse method: {dominant_wlen:.2f} km')
    print(f'Dominant angle by ellipse method: {dominant_theta:.0f} deg from north')

    # save to csv with results
    if not test:
        df = pd.read_excel('../sat_vs_ukv_results.xlsx', index_col=[0, 1])
        if not use_sim_sat:
            df.loc[(f'{s.year}-{s.month:02d}-{s.day:02d}', region), 'ukv_lambda_ellipse'] = dominant_wlen
            df.loc[(f'{s.year}-{s.month:02d}-{s.day:02d}', region), 'ukv_theta_ellipse'] = dominant_theta
        else:
            df.loc[(f'{s.year}-{s.month:02d}-{s.day:02d}', region), 'ukv_radsim_lambda_ellipse'] = dominant_wlen
            df.loc[(f'{s.year}-{s.month:02d}-{s.day:02d}', region), 'ukv_radsim_theta_ellipse'] = dominant_theta

        df.to_excel('../sat_vs_ukv_results.xlsx')

    np.save(save_path + 'pspec_array.npy', pspec_2d.data)

#     TODO save arrays so that they can be compared to eumetsat
# TODO put plotting stuff in another file? so that that won't slow down running of file
