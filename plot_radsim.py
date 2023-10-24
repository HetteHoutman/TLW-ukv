import sys

import netCDF4 as nc
import matplotlib.pyplot as plt
import iris
import numpy as np

from miscellaneous import check_argv_num

if __name__ == '__main__':
    check_argv_num(sys.argv, 1, '(radsim output file)')
    filename = sys.argv[1]

    varname = 'refl'

    output = nc.Dataset(filename)

    if varname not in output.variables:
        print('Error: varname ' + varname + ' not found in ' + filename)
        sys.exit(1)
    if 'lat' not in output.variables or 'lon' not in output.variables:
        print('Error: netCDF file must contain lat and lon values')
        sys.exit(1)

    refl = np.rot90(output.variables['refl'][:].reshape(621, 808, order='F'))
    plt.imshow(refl, cmap='gray')
    plt.colorbar()
    plt.title('Reflectance')
    plt.show()
