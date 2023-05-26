def data_from_pp_filename(filename):
    """
    # TODO look up proper name for forecast time
    Returns the year, month, dat and "forecast time" of a pp file
    Parameters
    ----------
    filename

    Returns
    -------

    """
    year = filename[-18:-14]
    month = filename[-14:-12]
    day = filename[-12:-10]
    forecast_time = filename[-9:-7]

    return year, month, day, forecast_time
