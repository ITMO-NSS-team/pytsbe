import os


def get_label_and_horizon(file_path: str, name_pattern: str):
    """ Parse path to the file and define time series label and forecast horizon

    :param file_path: absolute path to the file
    :param name_pattern: part of file name which does not contain any information
    about ts label and horizon
    """
    # Get forecast horizon and time series name
    current_folder = os.path.basename(file_path)
    ts_label_forecast_horizon = current_folder.split(name_pattern)[0]

    splitted = ts_label_forecast_horizon.split('_')
    if len(splitted) == 2:
        ts_label, forecast_horizon = splitted
    else:
        # Name of time series complex and contains '_'
        forecast_horizon = splitted[-1]
        ts_label = ts_label_forecast_horizon.split(f'_{forecast_horizon}')[0]

    return ts_label, int(forecast_horizon)
