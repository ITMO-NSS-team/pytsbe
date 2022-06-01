from pytsbe.data.exploration import DataExplorer


def explore_available_datasets():
    """
    Example of how to launch data exploration. For all datasets in data folder
    perform calculation of stationary and non-stationary time series and create
    visualisation of time series.
    """
    explorer = DataExplorer()
    explorer.display_statistics()

    # Have a look at time series
    explorer.visualise_series()


if __name__ == '__main__':
    explore_available_datasets()
