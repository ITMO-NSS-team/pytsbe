import pandas as pd

from pytsbe.data.data import TimeSeriesDatasets, MultivariateTimeSeriesDatasets
from pytsbe.paths import get_path_for_dummy_dataset


def test_long_format_data_loaded_correctly():
    """ Check if the dataset in long format are loaded into memory correctly """
    dummy_path = get_path_for_dummy_dataset()
    ts_dataset = TimeSeriesDatasets.setup_from_long_format(path=dummy_path)

    assert len(ts_dataset.time_series) == 3
    assert len(ts_dataset.time_series[0]) == 100
    assert ts_dataset.labels[0] == 'series_1'


def test_univariate_series_load_correctly():
    """ Load desired dataset and check if time series were processed correctly """
    dataset = TimeSeriesDatasets.configure_dataset_from_path(dataset_name='FRED')

    assert len(dataset.time_series) == 12
    assert len(dataset.labels) == 12
    assert isinstance(dataset.time_series[0], pd.DataFrame)


def test_multivariate_series_load_correctly():
    """
    For multivariate time series all series stored in dataframe.
    Labels the same as for univariate time series.
    """
    dataset = MultivariateTimeSeriesDatasets.configure_dataset_from_path(dataset_name='SSH')

    assert len(dataset.labels) == 25
    assert isinstance(dataset.dataframe, pd.DataFrame)
