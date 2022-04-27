from pytsbe.data.exploration import DataExplorer


def explore_available_datasets():
    explorer = DataExplorer()
    explorer.display_statistics()

    # Look at the time series
    explorer.visualise_series()


if __name__ == '__main__':
    explore_available_datasets()
