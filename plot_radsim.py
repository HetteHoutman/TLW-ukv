import sys

import matplotlib.pyplot as plt

from miscellaneous import check_argv_num
from prepare_radsim_array import get_refl

if __name__ == '__main__':
    check_argv_num(sys.argv, 1, '(radsim output file)')
    filename = sys.argv[1]

    refl = get_refl(filename)
    plt.imshow(refl, cmap='gray')
    plt.colorbar()
    plt.title('Reflectance')
    plt.show()
