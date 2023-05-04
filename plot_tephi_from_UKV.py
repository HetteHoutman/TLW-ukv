import matplotlib.pyplot as plt
import iris
import iris.quickplot as qplt
import numpy as np

from iris_read import *

indir = '/home/users/sw825517/Documents/ukv_data/'
filename = indir + 'prodm_op_ukv_20230419_12_000.pp'

cubes = iris.load(filename)

# so use read_variable function and thermodynamics to calculate dewpoint
# however, q and t have different model levels, so need to interpolate
# don't know how p clarks code does this, and i cant get interpolate to work...
# try again with clearer mind