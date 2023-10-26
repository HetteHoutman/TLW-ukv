import iris
from iris.analysis import Linear

from cube_processing import read_variable


def regrid_10m_wind_and_append(settings, pp_file):
    surft = read_variable(pp_file, 24, settings.h)
    u = read_variable(pp_file, 3225, settings.h)
    v = read_variable(pp_file, 3226, settings.h)

    u_rg = u.regrid(surft, Linear())
    v_rg = v.regrid(surft, Linear())

    u_rg.attributes['STASH'] = u_rg.attributes['STASH']._replace(item=209)
    v_rg.attributes['STASH'] = v_rg.attributes['STASH']._replace(item=210)

    iris.save([u_rg, v_rg], pp_file, append=True)


def regrid_10m_wind_and_save(settings, pp_file, target_file):
    """
    !not used
    assumes places of certain cubes in list of cubes from pp_file
    u10m at -4, v10m -2, surface temperature -6
    Parameters
    ----------
    settings
    pp_file
    target_file

    Returns
    -------

    """

    full_pp = iris.load(pp_file)

    surft = full_pp[-6]
    u = full_pp[-4]
    v = full_pp[-2]

    u_rg = u.regrid(surft, Linear())
    v_rg = v.regrid(surft, Linear())

    u_rg.attributes['STASH'] = u_rg.attributes['STASH']._replace(item=309)
    v_rg.attributes['STASH'] = v_rg.attributes['STASH']._replace(item=310)

    full_pp[-4] = u_rg
    full_pp[-2] = v_rg

    iris.save(full_pp, target_file)