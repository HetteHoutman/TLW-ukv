import os
import sys

import cartopy.crs as ccrs
import iris.cube
import iris.plot as iplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from iris.analysis.cartography import rotate_winds
from matplotlib import ticker, colors

from cube_processing import read_variable, cube_at_single_level, add_orography, add_true_latlon_coords, \
    create_latlon_cube
from fourier import *
from miscellaneous import check_argv_num, load_settings


def filtered_inv_plot(img, filtered_ft, Lx, Ly, latlon=None, inverse_fft=True):
    if inverse_fft:
        fig, (ax1, ax3) = plt.subplots(1, 2, sharey=True)
    else:
        fig, ax1 = plt.subplots(1, 1)

    xlen = img.shape[1]
    ylen = img.shape[0]

    if latlon:
        physical_extent = [latlon[0], latlon[2], latlon[1], latlon[3]]
        xlabel = 'Longitude'
        ylabel = 'Latitude'
    else:
        pixel_x = Lx / xlen
        pixel_y = Ly / ylen
        physical_extent = [-Lx / 2 - pixel_x / 2, Lx / 2 + pixel_x / 2, -Ly / 2 - pixel_y / 2, Ly / 2 + pixel_y / 2]
        xlabel = 'x distance / km'
        ylabel = 'y distance / km'

    ax1.imshow(img,
               extent=physical_extent,
               cmap='gray')
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)

    if inverse_fft:
        inv = np.fft.ifft2(filtered_ft.filled(fill_value=1))
        ax3.set_title(f'{min_lambda} km < lambda < {max_lambda} km')
        ax3.imshow(abs(inv),
                   extent=physical_extent,
                   cmap='gray')
    # save?
    plt.tight_layout()
    plt.savefig('plots/' + str(sys.argv[1]) + '/sat_plot.png', dpi=300)
    plt.show()


def plot_2D_pspec(bandpassed_pspec, Lx, Ly, wavelength_contours=None):
    xlen = bandpassed_pspec.shape[1]
    ylen = bandpassed_pspec.shape[0]

    fig2, ax2 = plt.subplots(1, 1)
    # TODO change to pcolormesh? might be useful in search for maximum
    max_k = xlen // 2 * 2 * np.pi / Lx
    max_l = ylen // 2 * 2 * np.pi / Ly
    pixel_k = 2 * max_k / xlen
    pixel_l = 2 * max_l / ylen
    recip_extent = [-max_k - pixel_k / 2, max_k + pixel_k / 2, -max_l - pixel_l / 2, max_l + pixel_l / 2]

    im = ax2.imshow(bandpassed_pspec.data, extent=recip_extent, interpolation='none',
                    norm=mpl.colors.LogNorm(vmin=bandpassed_pspec.min(), vmax=bandpassed_pspec.max()))

    if wavelength_contours:
        K, L, dist_array, thetas = recip_space(Lx, Ly, bandpassed_pspec.shape)
        wavelengths = 2 * np.pi / dist_array
        con = ax2.contour(K, L, wavelengths, levels=wavelength_contours, colors=['k'], linestyles=['--'])
        ax2.clabel(con)

    ax2.set_title('2D Power Spectrum')
    ax2.set_xlabel(r"$k_x$" + ' / ' + r"$\rm{km}^{-1}$")
    ax2.set_ylabel(r"$k_y$" + ' / ' + r"$\rm{km}^{-1}$")
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    fig2.colorbar(im, extend='both')
    plt.tight_layout()
    plt.savefig('plots/' + str(sys.argv[1]) + '/2d_pspec.png', dpi=300)
    plt.show()


def plot_radial_pspec(pspec_array, vals, theta_ranges, dom_wnum):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0, 1, len(pspec_array))))
    for i, pspec in enumerate(pspec_array):
        plt.loglog(vals, pspec, label=f'{theta_ranges[i]}' + r'$ \leq \theta < $' + f'{theta_ranges[i + 1]}')

    xlen = orig.shape[1]
    ylen = orig.shape[0]
    pixel_x = Lx / xlen
    pixel_y = Ly / ylen
    ymin = np.nanmin(np.array(pspec_array))
    ymax = np.nanmax(np.array(pspec_array))

    plt.vlines(2 * np.pi / 8, ymin, ymax, 'k', linestyles='--')
    # plt.vlines(2 * np.pi / min(Lx, Ly), ymin, ymax, 'k', linestyles='dotted')
    # plt.vlines(np.pi / max(pixel_y, pixel_x), ymin, ymax, 'k', linestyles='dotted')
    plt.vlines(2 * np.pi / min_lambda, ymin, ymax, 'k', linestyles='-.')
    plt.vlines(2 * np.pi / max_lambda, ymin, ymax, 'k', linestyles='-.')
    plt.vlines(dom_wnum, ymin, ymax, 'k')

    plt.title('1D Power Spectrum')
    plt.xlabel(r"$|\mathbf{k}|$" + ' / ' + r"$\rm{km}^{-1}$")
    plt.ylabel(r"$P(|\mathbf{k}|)$")
    plt.ylim(ymin, ymax)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('plots/' + str(sys.argv[1]) + '/radial_pspec.png', dpi=300)
    plt.show()


def plot_ang_pspec(pspec_array, vals, wavelength_ranges):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0, 1, len(pspec_array))))
    for i, pspec in enumerate(pspec_array):
        plt.plot(vals, pspec,
                 label=f'{wavelength_ranges[i]} km' + r'$ \leq \lambda < $' + f'{wavelength_ranges[i + 1]} km')

    ax = plt.gca()
    ax.set_yscale('log')
    plt.title('Angular power spectrum')
    plt.ylabel(r"$P(\theta)$")
    plt.xlabel(r'$\theta$ (deg)')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()


def plot_interp_contour(grid, interp_values):
    """Plots interpolated values on a contour plot. Colourscale is weird though"""
    con = plt.contourf(grid[:, :, 0], grid[:, :, 1], interp_values, locator=ticker.LogLocator())
    plt.colorbar(con, extend='both')
    plt.xlabel(r"$|\mathbf{k}|$" + ' / ' + r"$\rm{km}^{-1}$")
    plt.ylabel(r'$\theta$')
    plt.show()


def plot_interp_pcolormesh(wnum_bins_interp, theta_bins_interp, interp_values):
    """Plots interpolated values on a pcolormesh plot."""
    plt.pcolormesh(wnum_bins_interp, theta_bins_interp, interp_values,
                   norm=colors.LogNorm(vmin=pspec_2d.min(), vmax=pspec_2d.max()), )
    plt.colorbar(extend='both')
    plt.xlabel(r"$|\mathbf{k}|$" + ' / ' + r"$\rm{km}^{-1}$")
    plt.ylabel(r'$\theta$')
    plt.show()


def plot_pspec_polar(wnum_bins, theta_bins, radial_pspec_array, scale='linear', xlim=None):
    plt.pcolormesh(wnum_bins, theta_bins, radial_pspec_array, norm=mpl.colors.LogNorm())
    plt.xscale(scale)
    if xlim is not None:
        plt.xlim(xlim)
    plt.ylim(theta_bins[0], theta_bins[-1])

    plt.vlines(2 * np.pi / min_lambda, theta_bins[0], theta_bins[-1], 'k', linestyles='-.')
    plt.vlines(2 * np.pi / max_lambda, theta_bins[0], theta_bins[-1], 'k', linestyles='-.')

    plt.colorbar()
    plt.xlabel(r"$|\mathbf{k}|$" + ' / ' + r"$\rm{km}^{-1}$")
    plt.ylabel(r'$\theta$')


def plot_wind(w_cube, u_cube, v_cube, step=25):
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
    plt.show()


if __name__ == '__main__':
    # TODO clean and put in functions
    check_argv_num(sys.argv, 1, "(settings json file)")
    s = load_settings(sys.argv[1])

    if not os.path.exists('plots/' + str(sys.argv[1])):
        os.makedirs('plots/' + str(sys.argv[1]))

    w_cube = read_variable(s.reg_file, 150, s.h)
    u_cube = read_variable(s.reg_file, 2, s.h).regrid(w_cube, iris.analysis.Linear())
    v_cube = read_variable(s.reg_file, 3, s.h).regrid(w_cube, iris.analysis.Linear())
    orog_cube = read_variable(s.orog_file, 33, s.orog_h)

    add_orography(orog_cube, w_cube, u_cube, v_cube)

    w_single_level = cube_at_single_level(w_cube, s.map_height, coord='altitude', bottomleft=s.map_bottomleft,
                                          topright=s.map_topright)
    u_single_level = cube_at_single_level(u_cube, s.map_height, bottomleft=s.map_bottomleft, topright=s.map_topright)
    v_single_level = cube_at_single_level(v_cube, s.map_height, bottomleft=s.map_bottomleft, topright=s.map_topright)

    empty = create_latlon_cube(s)

    orig = w_single_level.regrid(empty, iris.analysis.Linear())
    u_rot, v_rot = rotate_winds(u_single_level, v_single_level, empty.coord_system())
    u_rot = u_rot.regrid(empty, iris.analysis.Linear())
    v_rot = v_rot.regrid(empty, iris.analysis.Linear())

    plot_wind(orig, u_rot, v_rot)

    add_true_latlon_coords(w_cube, u_cube, v_cube, orog_cube)

    Lx, Ly = extract_distances(orig.coords('latitude')[0].points, orig.coords('longitude')[0].points)
    orig = orig[0, ::-1].data

    K, L, wavenumbers, thetas = recip_space(Lx, Ly, orig.shape)
    wavelengths = 2 * np.pi / wavenumbers

    # orig = stripey_test(orig, Lx, Ly, [10, 5], [15, 135])

    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    min_lambda = 3
    max_lambda = 20
    bandpassed = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)
    filtered_inv_plot(orig, bandpassed, Lx, Ly, inverse_fft=True
                      # latlon=area_extent
                      )

    # TODO check if this is mathematically the right way of calculating pspec
    pspec_2d = np.ma.masked_where(bandpassed.mask, abs(shifted_ft) ** 2)
    plot_2D_pspec(pspec_2d.data, Lx, Ly, wavelength_contours=[5, 10, 35])

    wnum_bin_width = 0.1
    theta_bin_width = 5
    # noinspection PyTupleAssignmentBalance
    radial_pspec, wnum_bins, wnum_vals, theta_bins, theta_vals = make_polar_pspec(pspec_2d, wavenumbers, wnum_bin_width,
                                                                                  thetas, theta_bin_width)

    # radial_pspec *= wnum_vals**2

    bounded_polar_pspec, bounded_wnum_vals = apply_wnum_bounds(radial_pspec, wnum_vals, wnum_bins,
                                                               (min_lambda, max_lambda))
    dominant_wnum, dominant_theta = find_max(bounded_polar_pspec, bounded_wnum_vals, theta_vals)

    plot_pspec_polar(wnum_bins, theta_bins, radial_pspec)

    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    plt.show()
    plot_pspec_polar(wnum_bins, theta_bins, radial_pspec, scale='log', xlim=(0.05, 4.5))
    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    plt.savefig('plots/' + str(sys.argv[1]) + '/polar_pspec.png', dpi=300)
    plt.show()

    print(f'Dominant wavelength: {2 * np.pi / dominant_wnum:.2f} km')
    print(f'Dominant angle: {dominant_theta:.0f} deg from north')

    plot_radial_pspec(radial_pspec, wnum_vals, theta_bins, dominant_wnum)

    print('smoothing?')
