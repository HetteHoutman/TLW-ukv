import sys

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import netCDF4 as nc

from cube_processing import read_variable, cube_from_array_and_cube, create_latlon_cube
from miscellaneous import check_argv_num, load_settings, get_region_var


def get_refl(nc_file):
    varname = 'refl'
    output = nc.Dataset(nc_file)

    if varname not in output.variables:
        print('Error: varname ' + varname + ' not found in ' + nc_file)
        sys.exit(1)
    if 'lat' not in output.variables or 'lon' not in output.variables:
        print('Error: netCDF file must contain lat and lon values')
        sys.exit(1)

    refl = output.variables['refl'][:].reshape(808, 621)[::-1]
    return refl


if __name__ == '__main__':
    check_argv_num(sys.argv, 3, '(settings, region, radsim output file)')
    s = load_settings(sys.argv[1])
    filename = sys.argv[3]
    datetime = f'{s.year}-{s.month:02d}-{s.day:02d}_{s.h}'

    sat_bounds = get_region_var("sat_bounds", sys.argv[2], '/home/users/sw825517/Documents/tephiplot/regions/')
    sat_bl, sat_tr = sat_bounds[:2], sat_bounds[2:]

    map_bounds = get_region_var("map_bounds", sys.argv[2], '/home/users/sw825517/Documents/tephiplot/regions/')
    map_bl, map_tr = map_bounds[:2], map_bounds[2:]

    refl = get_refl(filename)
    surf_t = read_variable("/home/users/sw825517/Documents/ukv_data/ukv_20230419_12_000.pp", 24, 12)

    refl_cube = cube_from_array_and_cube(refl[::-1], surf_t, unit=1, std_name='toa_bidirectional_reflectance')

    empty = create_latlon_cube(sat_bl, sat_tr, n=501)
    refl_regrid = refl_cube.regrid(empty, iris.analysis.Linear())