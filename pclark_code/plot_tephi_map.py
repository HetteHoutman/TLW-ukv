import matplotlib.pyplot as plt
import os.path

import iris

import iris.plot as iplt
import iris.quickplot as qplt
import datetime
import time as time
from iris.time import PartialDateTime

import tephi

import cartopy
import cartopy.crs as ccrs

import numpy as np
import thermodynamics as th

import csv

import metpy
import metpy.calc as mpcalc
from metpy.units import units


class named_Tephigram(tephi.Tephigram):
    '''
    Defines sub-class of Tephigram 
    including a '.name' attrbute.
    '''
    def __init__(self, *args, **kwargs):
        save_name=kwargs['name']
        del kwargs['name']
        super(named_Tephigram,self).__init__(*args, **kwargs)
        self.name=save_name

    def __str__(self):
        rep = "Named Tephigram\n"
        rep += "name: " + self.name +"\n"+super(named_Tephigram,self).__str__()
        return rep


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
    print("Reading data from stash %d at hour %d"%(code, hour_selected))
    hour_const = iris.Constraint(time=lambda cell : cell.point.hour == hour_selected)
    cube = cubes.extract(hour_const)

    return cube

def iris_stash_code(code):
    '''
    Converts stash code to iris format
    
    Args:
        code : Stash code string of up to 5 digits
        
    Returns:
        stash code in iris format
    '''
    temp = "%05d" % int(code)
    iris_stash_code = 'm01s'+temp[0:2]+'i'+temp[2:]
    return iris_stash_code

def get_column(pr, u, v, T, pt, q, top_level, xpos, ypos):
    '''
    Extract a column from the nearest point to xpos, ypos 
    from 3D cubes, u, v, T, pt, q
    '''

    global xcoord_name, ycoord_name
    lon_index = np.where(T.coord(xcoord_name).points <= xpos)[0][-1]
    lat_index = np.where(T.coord(ycoord_name ).points  <= ypos)[0][-1]

    grid_index=(lon_index, lat_index)

    prcol = pr.data[0:top_level, lat_index, lon_index]
    prcol_hPa = prcol/100.0

    ucol=0.5*(u.data[0:top_level, lat_index, lon_index-1] + \
              u.data[0:top_level, lat_index, lon_index])
    vcol=0.5*(v.data[0:top_level, lat_index-1, lon_index] + \
              v.data[0:top_level, lat_index, lon_index])


    Tcol  = T.data[0:top_level, lat_index, lon_index]
    ptcol = pt.data[0:top_level, lat_index, lon_index]
    qcol  = q.data[0:top_level, lat_index, lon_index]

    TDcol=th.dewpoint(Tcol, ptcol, qcol)
    Tcol_C=Tcol-273.15

    TDcol_C=TDcol-273.15
    ptcol_hPa=ptcol/100.0

    if xcoord_name == 'longitude':

        true_lon = xpos
        true_lat = ypos

        true_u = ucol
        true_v = vcol

    else:

    # Now convert to true long/lat
#    crs_sphere=ccrs.Geodetic()
        crs_sphere  = ccrs.PlateCarree()
#    crs_rotated = T.coord(ycoord_name).coord_system.as_cartopy_crs()
        crs_rotated = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)

        true_lon, true_lat =crs_sphere.transform_point(xpos, ypos, crs_rotated)

        xcol=np.copy(ucol)
        xcol[:]=xpos

        ycol=np.copy(vcol)
        ycol[:]=ypos

        true_u, true_v = crs_sphere.transform_vectors(crs_rotated, xcol, ycol, ucol, vcol)

#    true_u, true_v = ucol, vcol

    # Speed in knots
    spdcol =np.sqrt(true_u*true_u + true_v*true_v)*1.94384449

    # Direction in degrees from N
    dircol = np.arctan2(true_u, true_v)*180/np.pi+180

#    print ucol, vcol

#    print spdcol, dircol

    true_lonlat = (true_lon, true_lat)


    Temps=list(zip(ptcol_hPa,Tcol_C))

    Dews=list(zip(ptcol_hPa,TDcol_C))

    Winds=list(zip(spdcol, dircol, prcol_hPa))



    return grid_index, true_lonlat, Temps, Dews, Winds

def profile_data(Temps, Dews, Winds, tpg):

    def get_ascent_info():
        strp = ""
        strp += f"P: {p_parcel:~4.2f}, T: {T_parcel:~3.1f}, TD: {TD_parcel:~3.1f} \n"
        strp += f"CAPE: {CAPE:~5.2f}, CIN: {CIN:~5.2f} \n"
        strp += f"LCL P:{lcl_pressure:~5.2f}, T: {lcl_temperature:~3.1f} \n"
        strp += f"LFC P:{lfc_pressure:~5.2f}, T: {lfc_temperature:~3.1f} \n"
        strp += f"EL P:{el_pressure:~5.2f}, T: {el_temperature:~3.1f} \n\n"
        return strp

    ptcol_hPa = [ d[0] for d in Temps ] * units.hPa

    Tcol_C = [ d[1] for d in Temps ] * units.degC

    TDcol_C = [ d[1] for d in Dews ] * units.degC

    surface_plotted = False


    p_parcel, T_parcel, TD_parcel, index_parcel = mpcalc.most_unstable_parcel(ptcol_hPa, Tcol_C, TDcol_C, depth=600 * units.hPa)

    strparcel = ""
    if index_parcel > 2 :
        strparcel += "Most Unstable Parcel\n"
    else :
        strparcel += "Surface and Most Unstable Parcel\n"
        surface_plotted = True

#    print(p_parcel, T_parcel, TD_parcel, index_parcel)

    CAPE , CIN = mpcalc.most_unstable_cape_cin(ptcol_hPa, Tcol_C, TDcol_C)

    lcl_pressure, lcl_temperature = mpcalc.lcl(ptcol_hPa[index_parcel], Tcol_C[index_parcel], TDcol_C[index_parcel])

    lfc_pressure, lfc_temperature = mpcalc.lfc(ptcol_hPa[index_parcel:], Tcol_C[index_parcel:], TDcol_C[index_parcel:], which = 'most_cape')
    el_pressure, el_temperature = mpcalc.el(ptcol_hPa[index_parcel:], Tcol_C[index_parcel:], TDcol_C[index_parcel:], which = 'most_cape')
    strparcel += get_ascent_info()
    p, Ta, TDa, Tp = mpcalc.parcel_profile_with_lcl(ptcol_hPa[index_parcel:], Tcol_C[index_parcel:], TDcol_C[index_parcel:])
    parcel = list(zip(p.to(units.hPa).magnitude, Tp.to(units.degC).magnitude))

#    print('Plotting parcel:',parcel)
    Tparcel = [tpg.plot(parcel)]

    if not surface_plotted:
        strparcel += "Surface Parcel\n"
        index_parcel = 0
        CAPE , CIN = mpcalc.surface_based_cape_cin(ptcol_hPa, Tcol_C, TDcol_C)
        lcl_pressure, lcl_temperature = mpcalc.lcl(ptcol_hPa[index_parcel], Tcol_C[index_parcel], TDcol_C[index_parcel])

        lfc_pressure, lfc_temperature = mpcalc.lfc(ptcol_hPa[index_parcel:], Tcol_C[index_parcel:], TDcol_C[index_parcel:])
        el_pressure, el_temperature = mpcalc.el(ptcol_hPa[index_parcel:], Tcol_C[index_parcel:], TDcol_C[index_parcel:])
        strparcel += get_ascent_info()

        p, Ta, TDa, Tp = mpcalc.parcel_profile_with_lcl(ptcol_hPa[index_parcel:], Tcol_C[index_parcel:], TDcol_C[index_parcel:])
        parcel = list(zip(p.to(units.hPa).magnitude, Tp.to(units.degC).magnitude))
        Tparcel.append(tpg.plot(parcel))
    print(strparcel)

    return Tparcel, strparcel

def gen_tephi(xpos, ypos, poslist=None):
    global pr, u, v, T, pt, q, top_level, tm, \
        xcoord_name, ycoord_name
#        T1, T2, plt1_ax, fig1, tpg, \


    if np.min(T.coord(xcoord_name).points) \
             <= xpos+360 \
             <= np.max(T.coord(xcoord_name).points) :
        xpos += 360

    if (   np.min(T.coord(xcoord_name).points) <= xpos \
        <= np.max(T.coord(xcoord_name).points) and \
           np.min(T.coord(ycoord_name ).points) <= ypos \
        <= np.max(T.coord(ycoord_name ).points)) :

        grid_index, true_lonlat, Temps, Dews, Winds = \
          get_column(pr, u, v, T, pt, q, top_level, xpos, ypos)



#        print(Temps)
#        print(Dews)
#        print(Winds)

        if poslist is not None:
            poslist.append(true_lonlat)

        time_str = f'{tm.year:4d}{tm.month:02d}{tm.day:02d}_{tm.hour:02d}{tm.minute:02d}'
        print('Lon: %+09.4f Lat: %+08.4f'%(xpos, ypos))
        print('Top level: %3d x index: %4d y index: %4d'%(top_level,
                                                          grid_index[0],
                                                          grid_index[1]))
        LonLatStr = '%+09.4f_%+08.4f'%true_lonlat
        FigTitle = time_str + ' Lon: %9.4f Lat: %8.4f'%true_lonlat
        print(FigTitle)

        fig1=plt.figure('Tephigram', figsize=(8,8))
        fig1.clf()

        tephi.ISOBAR_SPEC = [(50, None)]
        tephi.WET_ADIABAT_SPEC = [(5, None)]
        tephi.MIN_THETA = -40
        tephi.MAX_THETA = 120

        tpg = named_Tephigram(name="Tephi_"+ time_str +LonLatStr,
                              figure=fig1, anchor=[(1000, -20), (200, -20)])

        T1 = tpg.plot(Temps)
        T1.barbs(Winds)
        T2 = tpg.plot(Dews)

        Tparcel, strparcel = profile_data(Temps, Dews, Winds, tpg)

        plt.title(FigTitle)

        print('File str is ', tpg.name)
        with open(tpg.name+'.txt', 'w') as txtfile:
            txtfile.write(strparcel)
        fig1.savefig(tpg.name+'.png')

    return

def tephi_from_csv(filename):
    with open(filename, 'r',newline='') as csvfile:
        rdr = csv.reader(csvfile)
        next(rdr)
        for row in rdr:
            [xpos, ypos] = [float(item) for item in row]
            gen_tephi(xpos, ypos, poslist=poslist)
    return

def onclick(event):

    global poslist
    print(event)

    if event.inaxes != plt1_ax: return

    print('button=%d'%(event.button))

    # if event.button == 3:

    #     print('File str is ', tpg.name)
    #     fig1.savefig(tpg.name+'.png')
    #     return

#    if event.xdata != None and event.ydata != None:

#        print 'x_pixel=%d, y_pixel=%d, x_coord=%f, y_coord=%f'%(
#        event.button, event.x, event.y, event.xdata, event.ydata)

    xpos = event.xdata
    ypos = event.ydata

    print(xpos, ypos)
    gen_tephi(xpos, ypos, poslist=poslist)

    plt.show()

    return

indir = '/storage/silver/diamet/sws98slg/UG_elevated_conv_project/'

TW_data_dir = indir
TW_file_name = 'prods_op_gl-mn_20190723_18_000.pp'
TP_data_dir = indir
TP_file_name = 'prodm_op_gl-mn_20190723_18_000.pp'


h = 18
# Setup area for plotting
longbound = [-20,20] # Longitude boundaries
latbound  = [40,70]  # Latitude boundaries


p_rho_cube = read_variable(TP_data_dir+TP_file_name,407,h)
u_cube = read_variable(TP_data_dir+TP_file_name,2,h)
v_cube = read_variable(TP_data_dir+TP_file_name,3,h)
p_theta_cube = read_variable(TP_data_dir+TP_file_name,408,h)
T_cube = read_variable(TP_data_dir+TP_file_name,16004,h)
q_cube = read_variable(TP_data_dir+TP_file_name,10,h)


#print(p_theta_cube, T_cube, q_cube)


# Just select out the 18Z data over area required
pr=p_rho_cube[0].intersection(longitude=longbound,latitude=latbound)
u=u_cube[0].intersection(longitude=longbound,latitude=latbound)
v=v_cube[0].intersection(longitude=longbound,latitude=latbound)
T=T_cube[0].intersection(longitude=longbound,latitude=latbound)
pt=p_theta_cube[0].intersection(longitude=longbound,latitude=latbound)
q=q_cube[0].intersection(longitude=longbound,latitude=latbound)

xcoord_name = 'longitude'
ycoord_name = 'latitude'
tm=T.coord('time').units.num2date(T.coord('time').points)[0]

# I only want to plot up to 15 km - find out which model level this is
max_height=15000.0
level=np.where(T.coords('level_height')[0].points > max_height)
# The object returned my where appears to be a tuple of arrays.
top_level=level[0][0]

# Here's an example of plotting a known point.
# Herstmonceux
xpos = 0.3166
ypos = 50.8910

gen_tephi(xpos, ypos)

if True :
# Now chose points interactively

    wbpt_cube = read_variable(TW_data_dir+TW_file_name, 16205, h)


    wbpt=wbpt_cube[0].intersection(longitude=longbound, latitude=latbound)


    print("Selected time: ", wbpt.coord('time'))


    i = np.where(wbpt.coord('pressure').points == 850)[0][0]


    fig0=plt.figure('Map')

    print('Plotting Map')

    qplt.contourf(wbpt[i], 20)
    plt.gca().coastlines(resolution='10m',)

    # get the current axes' subplot for use later on
    poslist = []
    plt1_ax = plt.gca()
    #fig = plt.gcf()
    cid = fig0.canvas.mpl_connect('button_press_event', onclick)

    fig1=plt.figure('Tephigram', figsize=(8,8))

    plt.show()

    with open('tephi_points.csv', 'w', newline='') as csvfile:
        tephiwriter = csv.writer(csvfile)
        tephiwriter.writerow(('long', 'lat'))
        tephiwriter.writerows(poslist)

    print("Points list stored in file tephi_points.csv" )


