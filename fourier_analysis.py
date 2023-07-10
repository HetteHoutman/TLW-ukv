import sys

import matplotlib as mpl
import iris.cube
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import scipy.interpolate
from iris.analysis.cartography import rotate_winds
from matplotlib import ticker, colors
import iris.plot as iplt
import cartopy.crs as ccrs

from cube_processing import read_variable, get_grid_latlon_from_rotated, add_grid_latlon_to_cube, cube_at_single_level, \
    add_orography, add_true_latlon_coords
from miscellaneous import check_argv_num, load_settings
from scipy import stats
from scipy.signal import argrelmax

def ideal_bandpass(ft, Lx, Ly, low, high):
    _, _, dist_array, thetas = recip_space(Lx, Ly, ft.shape)

    low_mask = (dist_array < low)
    high_mask = (dist_array > high)
    masked = np.ma.masked_where(low_mask | high_mask, ft)

    return masked


def recip_space(Lx, Ly, shape):
    xlen = shape[1]
    ylen = shape[0]

    # np uses linear frequency f instead of angular frequency omega=2pi*f, so multiply by 2pi to get angular wavenum k
    k = 2 * np.pi * np.fft.fftfreq(xlen, d=Lx / xlen)
    l = 2 * np.pi * np.fft.fftfreq(ylen, d=Ly / ylen)

    # do fft shift
    K, L = np.meshgrid(np.roll(k, k.shape[0] // 2), np.roll(l, l.shape[0] // 2))

    dist_array = np.sqrt(K ** 2 + L ** 2)
    thetas = -np.rad2deg(np.arctan2(K, L)) + 180
    thetas %= 180
    return K, L, dist_array, thetas


def extract_distances(lats, lons):
    g = pyproj.Geod(ellps='WGS84')
    _, _, Lx = g.inv(lons[0], lats[lats.shape[0] // 2],
                     lons[-1], lats[lats.shape[0] // 2])
    _, _, Ly = g.inv(lons[lons.shape[0] // 2], lats[0],
                     lons[lons.shape[0] // 2], lats[-1])

    return Lx / 1000, Ly / 1000


def create_bins(range, bin_width):
    bins = np.linspace(range[0], range[1], int(np.ceil((range[1] - range[0]) / bin_width) + 1))
    vals = 0.5 * (bins[1:] + bins[:-1])
    return bins, vals

def stripey_test(orig_shape, Lx, Ly, wavelens, angles):
    x = np.linspace(-Lx / 2, Lx / 2, orig_shape[1])
    y = np.linspace(-Ly / 2, Ly / 2, orig_shape[0])
    X, Y = np.meshgrid(x, y)
    total = np.zeros(X.shape)

    for wavelen, angle in zip(wavelens, angles):
        total += make_stripes(X, Y, wavelen, angle)

    # this ensures the stripes are roughly in the same range as the input data
    middle = (orig.max() + orig.min()) / 2
    total *= (orig.max() - orig.min()) / (total.max() - total.min())
    total += middle

    return total


def make_radial_pspec(pspec_2d: np.ma.masked_array, wavenumbers, wavenumber_bin_width, thetas, theta_bin_width):
    # TODO get rid of angular pspec since this function essentially does the same?
    wnum_bins, wnum_vals = create_bins((0, wavenumbers.max()), wavenumber_bin_width)
    theta_ranges, theta_vals = create_bins((-theta_bin_width / 2, 180 - theta_bin_width / 2), theta_bin_width)
    thetas_redefined = thetas.copy()
    thetas_redefined[(180 - theta_bin_width / 2 <= thetas_redefined) & (thetas_redefined < 180)] -= 180
    radial_pspec_array = []

    for i in range(len(theta_ranges) - 1):
        low_mask = thetas_redefined >= theta_ranges[i]
        high_mask = thetas_redefined < theta_ranges[i + 1]
        mask = (low_mask & high_mask)

        radial_pspec, _, _ = stats.binned_statistic(wavenumbers[mask].flatten(), pspec_2d.data[mask].flatten(),
                                                    statistic="mean",
                                                    bins=wnum_bins)
        radial_pspec *= np.pi * (wnum_bins[1:] ** 2 - wnum_bins[:-1] ** 2) * np.deg2rad(theta_bin_width)
        radial_pspec_array.append(radial_pspec)

    return np.array(radial_pspec_array), wnum_bins, wnum_vals, theta_ranges, theta_vals


def make_angular_pspec(pspec_2d: np.ma.masked_array, thetas, theta_bin_width, wavelengths, wavelength_ranges):
    # TODO change pspec_2d to normal array not masked, as this is not needed
    theta_bins, theta_vals = create_bins((-theta_bin_width / 2, 180 - theta_bin_width / 2), theta_bin_width)
    thetas_redefined = thetas.copy()
    thetas_redefined[(180 - theta_bin_width / 2 <= thetas_redefined) & (thetas_redefined < 180)] -= 180
    ang_pspec_array = []
    for i in range(len(wavelength_ranges) - 1):
        low_mask = wavelengths >= wavelength_ranges[i]
        high_mask = wavelengths < wavelength_ranges[i + 1]
        mask = (low_mask & high_mask)

        ang_pspec, _, _ = stats.binned_statistic(thetas_redefined[mask].flatten(), pspec_2d.data[mask].flatten(),
                                                 statistic="mean",
                                                 bins=theta_bins)
        ang_pspec *= np.deg2rad(theta_bin_width) * (
                (2 * np.pi / wavelength_ranges[i]) ** 2 - (2 * np.pi / wavelength_ranges[i + 1]) ** 2
        )
        ang_pspec_array.append(ang_pspec)

    return ang_pspec_array, theta_vals

def make_stripes(X, Y, wavelength, angle):
    angle += 90
    angle = np.deg2rad(angle)
    return np.sin(2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength)


def interp_to_polar(pspec_2d, wavenumbers, thetas, theta_bins=(0, 180), theta_step=1, wnum_range=(0.2, 2), wnum_step=0.01):
    """Interpolates power spectrum onto polar grid"""

    # create values of theta and wavenumber at which to interpolate
    theta_bins_interp, theta_gridp = create_bins(theta_bins, theta_step)
    wnum_bins_interp, wavenumber_gridp = create_bins(wnum_range, wnum_step)
    meshed_polar = np.meshgrid(wavenumber_gridp, theta_gridp)

    points = np.array([[k, l] for k, l in zip(wavenumbers.flatten(), thetas.flatten())])
    xi = np.array([[w, t] for w, t in zip(meshed_polar[0].flatten(), meshed_polar[1].flatten())])
    values = pspec_2d.flatten()

    interp_values = scipy.interpolate.griddata(points, values.data, xi, method='linear')

    grid = xi.reshape(meshed_polar[0].shape[0], meshed_polar[0].shape[1], 2)

    return wnum_bins_interp, theta_bins_interp, grid, interp_values.reshape(meshed_polar[0].shape)


def find_max(polar_pspec, wnum_vals, theta_vals):
    meshed_polar = np.meshgrid(wnum_vals, theta_vals)
    dom_theta_idx, dom_wnums_idx = argrelmax(polar_pspec)
    max_idx = np.nanargmax(polar_pspec)
    #
    return meshed_polar[0].flatten()[max_idx], meshed_polar[1].flatten()[max_idx]
    # return wnum_vals[dom_wnums_idx], theta_vals[dom_theta_idx]


def apply_wnum_bounds(polar_pspec, wnum_vals, wnum_bins, wlen_range):
    min_mask = (wnum_bins > 2*np.pi/wlen_range[1])[:-1]
    max_mask = (wnum_bins < 2*np.pi/wlen_range[0])[1:]
    mask = (min_mask & max_mask)

    return polar_pspec[:, mask], wnum_vals[mask]


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
    plt.savefig('plots/sat_plot.png', dpi=300)
    plt.show()


def plot_2D_pspec(bandpassed_pspec, Lx, Ly, wavelength_contours=None):
    xlen = bandpassed_pspec.shape[1]
    ylen = bandpassed_pspec.shape[0]

    fig2, ax2 = plt.subplots(1, 1)
    # TODO change to pcolormesh?
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
    plt.savefig('plots/2d_pspec.png', dpi=300)
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
    plt.savefig('plots/radial_pspec.png', dpi=300)
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


def create_latlon_cube():
    n = 500
    lat_bounds = [s.satellite_bottomleft[1], s.satellite_topright[1]]
    lon_bounds = [s.satellite_bottomleft[0], s.satellite_topright[0]]
    lat_coord = iris.coords.DimCoord(np.linspace(*lat_bounds, n), standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord(np.linspace(*lon_bounds, n), standard_name='longitude', units='degrees')
    empty = iris.cube.Cube(np.empty((n, n)), dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)])
    new_cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    empty.coord(axis='x').coord_system = new_cs
    empty.coord(axis='y').coord_system = new_cs

    return empty


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
    check_argv_num(sys.argv, 1, "(settings json file)")
    s = load_settings(sys.argv[1])

    w_cube = read_variable(s.reg_file, 150, s.h)
    u_cube = read_variable(s.reg_file, 2, s.h).regrid(w_cube, iris.analysis.Linear())
    v_cube = read_variable(s.reg_file, 3, s.h).regrid(w_cube, iris.analysis.Linear())
    orog_cube = read_variable(s.orog_file, 33, s.orog_h)

    w_cube, u_cube, v_cube = add_orography(orog_cube, w_cube, u_cube, v_cube)

    w_single_level = cube_at_single_level(w_cube, s.map_height, coord='altitude', bottomleft=s.map_bottomleft, topright=s.map_topright)
    u_single_level = cube_at_single_level(u_cube, s.map_height, bottomleft=s.map_bottomleft, topright=s.map_topright)
    v_single_level = cube_at_single_level(v_cube, s.map_height, bottomleft=s.map_bottomleft, topright=s.map_topright)

    empty = create_latlon_cube()

    orig = w_single_level.regrid(empty, iris.analysis.Linear())
    u_rot, v_rot = rotate_winds(u_single_level, v_single_level, empty.coord_system())
    u_rot = u_rot.regrid(empty, iris.analysis.Linear())
    v_rot = v_rot.regrid(empty, iris.analysis.Linear())

    # TODO clean and put in functions
    plot_wind(orig, u_rot, v_rot)

    add_true_latlon_coords(w_cube, u_cube, v_cube, orog_cube)

    Lx, Ly = extract_distances(orig.coords('latitude')[0].points, orig.coords('longitude')[0].points)
    orig = orig[0, ::-1].data

    K, L, wavenumbers, thetas = recip_space(Lx, Ly, orig.shape)
    wavelengths = 2 * np.pi / wavenumbers

    # orig = stripey_test(orig.shape, Lx, Ly, [10, 5], [15, 135])

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
    plot_2D_pspec(pspec_2d, Lx, Ly, wavelength_contours=[5, 10, 35])

    wnum_bin_width = 0.1
    theta_bin_width = 5
    radial_pspec, wnum_bins, wnum_vals, theta_bins, theta_vals = make_radial_pspec(pspec_2d, wavenumbers, wnum_bin_width,
                                                            thetas, theta_bin_width)

    # radial_pspec *= wnum_vals**2

    bounded_polar_pspec, bounded_wnum_vals = apply_wnum_bounds(radial_pspec, wnum_vals, wnum_bins, (min_lambda, max_lambda))
    dominant_wnum, dominant_theta = find_max(bounded_polar_pspec, bounded_wnum_vals, theta_vals)

    plot_pspec_polar(wnum_bins, theta_bins, radial_pspec)

    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    plt.show()
    plot_pspec_polar(wnum_bins, theta_bins, radial_pspec, scale='log', xlim=(0.05, 4.5))
    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    plt.savefig('plots/polar_pspec.png', dpi=300)
    plt.show()

    print(f'Dominant wavelength: {2*np.pi / dominant_wnum:.2f} km')
    print(f'Dominant angle: {dominant_theta:.0f} deg from north')

    plot_radial_pspec(radial_pspec, wnum_vals, theta_bins, dominant_wnum)

    print('smoothing?')




