import sys

import matplotlib.pyplot as plt
import pandas as pd

from met_functions import N_squared, scorer_param


def plot_profile(l2, height, N2U2, theta, wind, direction):
  fig, (ax, ax3) = plt.subplots(1,2, sharey=True)
  ax2 = ax.twiny()
  ax4 = ax3.twiny()

  line_l2 = ax2.plot(l2, height, color='tab:red', linestyle='-', label='l^2')
  line_N2U2 = ax2.plot(N2U2, height, color='tab:red', linestyle='--', label='N^2/U^2')
  line_theta = ax.plot(theta, height, color='tab:blue', label='theta')
  line_wind = ax3.plot(wind, height, color='tab:purple', label='wind speed')
  line_dir = ax4.scatter(direction, height, color='tab:olive', label='wind direction')

  lines = line_l2 + line_N2U2 + line_theta
  labels = [l.get_label() for l in lines]
  ax.legend(lines, labels, loc='best')

  ax.set_xlabel('Potential temperature (K)')
  ax.set_ylabel('Altitude (m)')
  ax2.set_xlabel('l^2 (m^-2)')
  ax3.set_xlabel('Wind speed (m/s)')
  ax4.set_xlabel('Wind direction (degrees)')
  return fig



if __name__ == '__main__':

  txtfile = sys.argv[1]

  station = txtfile[-20:-15]
  year = txtfile[-14:-12]
  month = txtfile[-12:-10]
  day = txtfile[-10:-8]
  time = txtfile[-7:-4]

  print(f'Showing sounding from station {station} on {year}/{month}/{day} at {time}.')

  # read file
  data = pd.read_csv(txtfile, skiprows=[0,2,3], delim_whitespace=True)
  data = data[:100]
  data['SMPS'] = data.SKNT * 1.852 / 3.6
  data.drop('SKNT', axis=1, inplace=True)

  theta = data.THTA.values
  height = data.HGHT.values
  wind = data.SMPS.values

  # calculate N squared
  N2 = N_squared(theta, height)

  # include wind gradient in l squared
  # should really be the wind perpendicular to wavenumber vector, not implemented yet
  N2U2 = N2 / data.SMPS.values ** 2

  l2 = scorer_param(N2, wind, height)

  # plot
  mask = (height<5000)
  fig = plot_profile(l2[mask], height[mask], N2U2[mask], theta[mask], wind[mask], data.DRCT.values[mask])
  title = f'Station {station}, {year}/{month}/{day} {time}'
  plt.suptitle(title)

  plt.tight_layout()
  plt.savefig(f'plots/profile_from_txt_{station}_{year}{month}{day}_{time}.png', dpi=300)
  plt.show()

