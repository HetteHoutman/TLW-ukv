import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import iris

import iris.plot as iplt
import iris.quickplot as qplt

import numpy as np

import thermodynamics as th

def read_variable(pp_file, code, hour_selected):
    '''
    Reads variable defined by stash code from pp_file.
    
    Args:
        pp_file (str)
        code (int)
        
    Returns:
        cubes (list)
    '''
    stash_code=iris_stash_code(code)
    stash_const=iris.AttributeConstraint(STASH = stash_code)
    cubes = iris.load(pp_file, stash_const) 
    print(f"Reading data from stash {code:d} at hour {hour_selected:d}")
    hour_const = iris.Constraint(time=lambda cell : 
                                 cell.point.hour == hour_selected)
    cube = cubes.extract(hour_const)[0]
    
    return cube
    
def iris_stash_code(code):
    '''
    Converts stash code to iris format
    
    Args:
        code : Stash code string of up to 5 digits
        
    Returns:
        stash code in iris format
    '''
    temp = f"{code:05d}"
    iris_stash_code = 'm01s'+temp[0:2]+'i'+temp[2:]
    return iris_stash_code


def xsect_fig(fignum, cross_section, rh_cross_section, xcoord_name, ttle, figname, cminmax) :
    '''
    This is just a tarted up contour plot to save repetition.
    '''
#    print(cross_section)
    fig = plt.figure(fignum,figsize=(15, 6))
    fig.clf() 
    # Create an Axes, specifying the map projection.
 
    con=iplt.contourf(cross_section, np.arange(cminmax[0],cminmax[1],1), 
                      coords=[xcoord_name, 'air_pressure'])
    con1=iplt.contour(cross_section, np.arange(cminmax[0],cminmax[1],5), 
                      colors='k', coords=[xcoord_name, 'air_pressure'])
    ax1 = plt.gca()
    ax1.clabel(con1, inline=1, fontsize=10, fmt='%3.0f' )
    con2=iplt.contour(rh_cross_section, np.arange(80,110,10), 
                      colors='w', coords=[xcoord_name, 'air_pressure'])
    ax1.set_xlabel(xcoord_name)
    ax1.set_ylabel(r'Pressure/Pa')
    ax2 = plt.gca()
    ax2.clabel(con2, inline=1, fontsize=10, fmt='%3.0f' )
    plt.ylim([100000,30000])

# add a colourbar with a label
#    cbar = plt.colorbar(con, colorbar_axes, orientation='horizontal')
    cbar = plt.colorbar(con, orientation='vertical')
    cbar.set_label(cross_section.units)
    plt.title(ttle)

    plt.savefig(figname)
    plt.close()
 
    return con

def get_grid_latlon_from_rotated(cube):
# in the form of a cartopy map projection.
    crs_rotated = cube.coord('grid_latitude').coord_system.as_cartopy_crs()
    
    crs_sphere  = ccrs.PlateCarree()

    r_lats=cube.coord('grid_latitude').points.copy()
    r_lons=cube.coord('grid_longitude').points.copy()
    
    rot_lons, rot_lats = np.meshgrid(r_lons, r_lats)
    
    true_lons = np.zeros_like(theta.data[0,:,:])
    true_lats = np.zeros_like(theta.data[0,:,:])
    
    for i, r_lon in enumerate(r_lons):
        for j, r_lat in enumerate(r_lats):

            true_lons[j,i], true_lats[j,i] =crs_sphere.transform_point(r_lon, 
                                                        r_lat, crs_rotated)
    
    
    true_lons[true_lons>180] -= 360
    
    grid_latlon = {'rot_lons': rot_lons,
                   'rot_lats': rot_lats,
                   'true_lons': true_lons,
                   'true_lats': true_lats,
                   'lon_coord': iris.coords.AuxCoord(points=true_lons, 
                                            standard_name='longitude'),
                   'lat_coord': iris.coords.AuxCoord(points=true_lats, 
                                            standard_name='latitude')
                   }
    return grid_latlon

def get_coord_index(cube, name):
    for i, c in enumerate(cube.coords()):
        if name in c.standard_name :
            break
    return i

def add_pressure_to_cube(cube, pcoord):
    
    ilev = get_coord_index(cube,'model_level_number')
    ilat = get_coord_index(cube,'latitude')
    ilon = get_coord_index(cube,'longitude')
    cube.add_aux_coord(pcoord, [ilev, ilat, ilon])
    
          
def add_grid_latlon_to_cube(cube, grid_latlon):
    
    ilat = get_coord_index(cube,'grid_latitude')
    ilon = get_coord_index(cube,'grid_longitude')
    cube.add_aux_coord(grid_latlon['lat_coord'], [ilat, ilon])
    cube.add_aux_coord(grid_latlon['lon_coord'], [ilat, ilon])
    
 
############################### Enter settings here ##################
indir = '/storage/silver/diamet/sws98slg/UG_elevated_conv_project/'

ukv = True
#ukv = False
# Just select out the 18Z data
h = 18

#    lon_vals = np.linspace(-6,0,13)
#    lat_vals = np.linspace(49,52,7)
    
lon_vals = np.array([-1.4766, -1.6356])
lat_vals = np.array([51.3281, 51.7900])

if ukv :
    source = 'UKV'
    filename = indir + 'prodm_op_ukv_20190723_15_*.pp'
    
    p_theta_cube = read_variable(filename,408,h)
    T_cube = read_variable(filename,16004,h)
    q_cube = read_variable(filename,10,h)
    
    # Just select out the 18Z data over area required
    T = T_cube
    p = p_theta_cube
    q = q_cube[+1:,:,:]

    theta_data=th.potential_temperature(T.data, p.data)
    theta=T.copy()
    theta.data=theta_data
    theta.rename('potential temperature')
    
    grid_latlon = get_grid_latlon_from_rotated(theta)
    
    add_grid_latlon_to_cube(theta, grid_latlon)
    add_grid_latlon_to_cube(T, grid_latlon)
    add_grid_latlon_to_cube(p, grid_latlon)
    add_grid_latlon_to_cube(q, grid_latlon)
        
    (ny, nx) = np.shape(grid_latlon['true_lons'])
            
# Create list of required cross section indices.

    lats_1D = grid_latlon['true_lats'][:, nx//2]
    lons_1D = grid_latlon['true_lons'][ny//2, :]
    
    crs_from = theta.coord('grid_latitude').coord_system.as_cartopy_crs()
    
    model_lat=theta.coord('grid_latitude').points.copy()
    model_lon=theta.coord('grid_longitude').points.copy()

else:
    
    source = 'Global'
    filename = indir+'prodm_op_gl-mn_20190723_18_000.pp'
    
    # Setup area for plotting
    longbound = [-11,3] # Longitude boundaries
    latbound  = [48,60]  # Latitude boundaries
        
    p_theta_cube = read_variable(filename,408,h)
    theta_cube = read_variable(filename,4,h)
    q_cube = read_variable(filename,10,h)
    
    # Just select out the 18Z data over area required
    theta = theta_cube.intersection(longitude=longbound,latitude=latbound)
    p=p_theta_cube.intersection(longitude=longbound,latitude=latbound)
    q=q_cube.intersection(longitude=longbound,latitude=latbound)
    Tm=th.temperature(theta.data, p.data)
    T=theta.copy()
    T.data=Tm
    T.rename('temperature')
    
    lats_1D = theta.coord('latitude').points.copy()
    lons_1D = theta.coord('longitude').points.copy()
    
    crs_from  = ccrs.PlateCarree()
    
    model_lat = theta.coord('latitude').points.copy()
    model_lon = theta.coord('longitude').points.copy()


lon_indices = []
for lon_val in lon_vals:
    d = np.abs(lon_val - lons_1D)
    index = np.where(d == np.min(d))[0][0]
    lon_indices.append(index)
lat_indices = []
for lat_val in lat_vals:
    d = np.abs(lat_val - lats_1D)
    index = np.where(d == np.min(d))[0][0]
    lat_indices.append(index)
    
lons = lons_1D[lon_indices]
lats = lats_1D[lat_indices]
lons[lons>180] -= 360

# Time
tm=theta.coord('time').units.num2date(T.coord('time').points)[0]
time_str = f'{tm.year:4d}{tm.month:02d}{tm.day:02d}_{tm.hour:02d}{tm.minute:02d}'

# Create a new vertical coordinate from the appropriate pressure data.

pcoord = iris.coords.AuxCoord(points=p.data,\
  standard_name=p.standard_name,units=p.units)

# Add the pressure vertical coordinate (3D field) to T and q as aux_coords

add_pressure_to_cube(theta, pcoord)
add_pressure_to_cube(T, pcoord)
add_pressure_to_cube(q, pcoord)


# th_e=th.equiv_potential_temperature_accurate(T.data, p.data, q.data)
# theta_e=T.copy()
# theta_e.data=th_e
# theta_e.rename('equivalent potential temperature')

# th_es=th.equiv_potential_temperature_accurate(T.data, p.data, th.qsat(T.data, p.data))
# theta_es=T.copy()
# theta_es.data=th_es
# theta_es.rename('saturated equivalent potential temperature')

qs = th.qsat(T.data, p.data)
rh=T.copy()
rh.data=q.data/qs*100
rh.rename('RH')
rh.units='percent'

th_w=th.wet_bulb_potential_temperature(T.data, p.data, q.data)
theta_w=T.copy()
theta_w.data=th_w
theta_w.rename('wet-bulb potential temperature')

th_s=th.wet_bulb_potential_temperature(T.data, p.data, qs)
theta_s=T.copy()
theta_s.data=th_s
theta_s.rename('saturated wet-bulb potential temperature')


fig3 = plt.figure(3,figsize=(15, 10))    
fig3.clf() 
#plt.axes(projection=ccrs.LambertConformal())
crs_ps = ccrs.Stereographic()
plt.axes(projection=crs_ps)


conmap=iplt.contourf(theta_w[0,:,:],np.arange(17)+284)
plt.gca().coastlines(resolution='10m')
cbarmap = plt.colorbar(conmap, orientation='vertical')
plt.title(r'$\theta_w$ at surface')
# get the current axes
plt1_ax = plt.gca()

# This lat/long rotated grid - unfortunately draw_labels=True isn't implemented for plotting on rotated grids. 
# This sets up a standard 'Lat/Long' coordinate system
crs_latlon = ccrs.PlateCarree()

# Comment out to lose grid lines
plt1_ax.gridlines(crs=crs_latlon, linestyle='--', linewidth=1)

# Overplot lines on map.
for lat_index in lat_indices:
    x = np.zeros_like(model_lon)
    y = np.zeros_like(model_lon)
    for i, lon in enumerate(model_lon):
        x[i], y[i] = crs_ps.transform_point(lon, model_lat[lat_index], crs_from)
    plt.plot(x,y)
    
for lon_index in lon_indices:
    x = np.zeros_like(model_lat)
    y = np.zeros_like(model_lat)
    for i, lat in enumerate(model_lat):
        x[i], y[i] = crs_ps.transform_point(model_lon[lon_index], lat, crs_from)
    plt.plot(x,y)

# Save the map and show it.
plt.savefig(source+'_theta_w_map.png')
plt.show()

xcoord_name = 'longitude'
leader = source + time_str +'_'
for i, lat_index in enumerate(lat_indices):
    indstr=f'Latitude_{lats[i]:03.4f}'
    tail='_cross_sect_' + indstr + '.png'
    loc = f'Latitude {lats[i]:3.4f}'
    xsect_fig(0,theta[:,lat_index,:], rh[:,lat_index,:], xcoord_name, 
              r'$\theta$ '+loc , leader+'Theta'+tail,(280,331))

    xsect_fig(1,theta_w[:,lat_index,:], rh[:,lat_index,:], xcoord_name, 
              r'$\theta_w$ '+loc , leader+'Theta_w'+tail,(285,311))

    xsect_fig(2,theta_s[:,lat_index,:], rh[:,lat_index,:], xcoord_name, 
              r'${\theta}_s$ '+loc , leader+'Theta_s'+tail,(285,311))

ycoord_name = 'latitude'
for i, lon_index in enumerate(lon_indices):
    indstr=f'Longitude_{lons[i]:03.4f}'
    tail='_cross_sect_' + indstr + '.png'
    loc = f'Longitude {lons[i]:3.4f}'
    xsect_fig(0,theta[:,:,lon_index], rh[:,:,lon_index], ycoord_name, 
              r'$\theta$ '+loc , leader+'Theta'+tail,(280,331))

    xsect_fig(1,theta_w[:,:,lon_index], rh[:,:,lon_index], ycoord_name, 
              r'$\theta_w$ '+loc , leader+'Theta_w'+tail,(285,311))

    xsect_fig(2,theta_s[:,:,lon_index], rh[:,:,lon_index], ycoord_name, 
              r'${\theta}_s$ '+loc , leader+'Theta_s'+tail,(285,311))

