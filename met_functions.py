# this file contains some useful functions for meteorological calculations
import numpy as np
import thermodynamics_constants as tc

def uv_to_spddir(u, v):
    """
    converts u and v wind fields to wind speed and direction. Taken from Peter Clark's code.
    Parameters
    ----------
    u : float or int or ndarray
        x-component of wind
    v : float or int or ndarray
        y-component of wind

    Returns
    -------
    tuple
        tuple containing the absolute wind speed and direction (met. convention)
    """
    return np.sqrt(u**2 + v**2),  np.arctan2(u, v)*180/np.pi+180


def N_squared(theta, height):
  """
  calculates the brunt-vaisala frequency
  Parameters
  ----------
  theta : ndarray
    1d array containg theta values
  height : ndarray
    the corresponding vertical height coordinates

  Returns
  -------
  ndarray
    containing B-V freq. values

  """

  dthetadz = np.gradient(theta, height)
  N2 = tc.g / theta * dthetadz
  return N2

def scorer_param(N2, wind, height):
  """
  calculates the scorer parameter. does not currently take into account wind direction
  Parameters
  ----------
  N2 : ndarray
    1d array containing brunt-vaisala frequency values
  wind : ndarray
    1d array containing wind speed values
  height : ndarray
    corresponding height coordinates

  Returns
  -------

  """
  dudz = np.gradient(wind, height)
  d2udz2 = np.gradient(dudz, height)
  return N2 / wind ** 2 - d2udz2 / wind