import matplotlib.pyplot as plt
#import cartopy.crs as ccrs

import iris

import iris.plot as iplt
import iris.quickplot as qplt

import numpy as np
#from matplotlib import ticker, cm
#from matplotlib.colors import BoundaryNorm
#from matplotlib.ticker import MaxNLocator



indir = '/storage/silver/diamet/sws98slg/UG_elevated_conv_project/'
# Use this form to get all the files.
#filename = indir + 'prodm_op_ukv_20190723_15_*.pp'
#filename = indir + 'prods_op_ukv_20190723_15_002.pp'
#filename = indir + 'prods_op_gl-mn_20190723_18_*.pp'
filename = indir + 'prods_op_gl-mn_20190723_18_003.pp'
print(filename)

# Load whole file to look what's inside
#cubes=iris.load(filename)
#print(cubes)

# Setup area for plotting
longbound = [-10,5] # Longitude boundaries
latbound  = [45,60]  # Latitude boundaries


# Load stratiform_rainfall_flux - note load returns a list of cubes even if it 
# only has one member, so we select the [0]th.
# Change numbers to mm/h.

str_pptn_cube = iris.load(filename, 'stratiform_rainfall_flux')[0]
cnv_pptn_cube = iris.load(filename, 'convective_rainfall_flux')[0]

pptn = str_pptn_cube.intersection(longitude=longbound,latitude=latbound) + \
       cnv_pptn_cube.intersection(longitude=longbound,latitude=latbound) 
pptn.convert_units('kg m-2 h-1')
pptn.units = 'mm/h'
print(pptn)

# Rainfall rate contour values.!
pptn_cv = [0.01, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
pptn_colors=['#0000A0', '#5E5EFF', '#838300', '#FFD400', '#FF9500', 
             '#FF0000', '#FF00FF', '#FFFFFF']

# We need times of field for plot titles.
time_coord = pptn.coord('time')

# Choose an insipid colour map. 
# cmap = plt.get_cmap('PiYG')

# Loop over first index of pptn cube (time)
for it in range(np.shape(pptn)[0]) :
    print(time_coord[it].units.num2date(time_coord.points[it]))
# Create a figure
    fig = plt.figure(1,figsize=(6, 10))
# Creates a subplot
#    ax = fig.subplots()

# Filled contour plot of rainfall rate. Use log intervals in contour values.
    con=qplt.contourf(pptn[it,...], pptn_cv,  colors=pptn_colors)

# Add coastlines.
    plt.gca().coastlines(resolution='10m')
# Add time and date as title.
    tm = time_coord[it].units.num2date(time_coord.points[it])
    title=f'{tm.day:02d} {tm.hour:02d}:{tm.minute:02d}'
    plt.title(title)
# Display the plot. Need to close the plot window to carry on.
    fn = f'global_rain_{tm.day:02d}{tm.hour:02d}{tm.minute:02d}.png'
    plt.savefig(fn)
    plt.show()

