import matplotlib.pyplot as plt
import thermodynamics as th
from iris_read import *
from plot_profile_from_txt import plot_profile, N_squared, scorer_param
import datetime as dt

indir = '/home/users/sw825517/Documents/ukv_data/'
filename = indir + 'prodm_op_ukv_20150414_09_004.pp'

h = 12
p_rho_cube = read_variable(filename, 407, h)
u_cube = read_variable(filename, 2, h)
v_cube = read_variable(filename, 3, h)
p_theta_cube = read_variable(filename, 408, h)
T_cube = read_variable(filename, 16004, h)
q_cube = read_variable(filename, 10, h)

# only plot up to certain height
min_height = 150
max_height = 5000
height = u_cube.coord('level_height').points
level_mask = (height < max_height) & (height > min_height)
height = height[level_mask]

# calculate theta
theta_col = th.potential_temperature(T_cube.data[level_mask, 300, 400], p_theta_cube.data[level_mask, 300, 400])

# need to figure out how to get proper coordinates and how to rotate
u_col = u_cube.data[level_mask, 300, 400]
v_col = v_cube.data[level_mask, 300, 400]

spd_col = np.sqrt(u_col**2 + v_col**2)
dir_col = np.arctan2(u_col, v_col)*180/np.pi+180

# N squared
N2 = N_squared(theta_col, height)
N2U2 = N2 / u_col ** 2
l2 = scorer_param(N2, u_col, height)

fig = plot_profile(l2, height, N2U2, theta_col, spd_col, dir_col)
year = filename[-18:-14]
month = filename[-14:-12]
day = filename[-12:-10]
forecast_time = filename[-9:-7]

title = f'UKV [location] at {year}/{month}/{day} at {h} ({forecast_time})'
plt.suptitle(title)
plt.tight_layout()
plt.show()