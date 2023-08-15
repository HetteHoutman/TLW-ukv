import pygrib
import iris
import iris.plot as iplt
import numpy as np
import matplotlib.pyplot as plt


filename = "/home/users/sw825517/Documents/sat_ukv_comp/data/MSG3-SEVI-MSGCLTH-0100-0100-20230419120000.000000000Z-20230419121431-4854768.grb"
grbs = pygrib.open(filename)
grb = grbs.read(1)[0]

cth = iris.cube.Cube(grb.values[:, ::-1])
lats, lons = grb.latlons()
lon_coord = iris.coords.AuxCoord(points=lons, standard_name='longitude')
lat_coord = iris.coords.AuxCoord(points=lats, standard_name='latitude')

cth.add_aux_coord(lon_coord, [0, 1])
cth.add_aux_coord(lat_coord, [0, 1])

from miscellaneous import load_settings

s = load_settings("/home/users/sw825517/Documents/tephiplot/settings/20230419_12.json")

lat_mask = (s.satellite_bottomleft[1] < lats) & (lats < s.satellite_topright[1])
lon_mask = (s.satellite_bottomleft[0] < lons) & (lons < s.satellite_topright[0])

cth.data[~(lat_mask & lon_mask)] = np.nan


def index_selector_2d(desired, array):
    dists1 = np.abs(array[0] - desired[0])
    dists2 = np.abs(array[1] - desired[1])
    dists = dists1 * dists1 + dists2 * dists2
    return np.unravel_index(dists.argmin(), dists.shape)


lat_idx, lon_idx = index_selector_2d(s.gc_end, (cth.coord('latitude').points, cth.coord('longitude').points))
print(cth[lat_idx, lon_idx].data)

iplt.contourf(cth, levels=np.arange(160, 5000, 320), cmap='rainbow')
plt.colorbar()
ax = plt.gca()
ax.coastlines()
plt.xlim(s.satellite_bottomleft[0], s.satellite_topright[0])
plt.ylim(s.satellite_bottomleft[1], s.satellite_topright[1])
plt.show()
