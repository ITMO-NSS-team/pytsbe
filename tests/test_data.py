import pandas as pd
from datetime import datetime

from pytsbe.data.data import TimeSeriesDatasets, MultivariateTimeSeriesDatasets, dataclass_by_name
from pytsbe.paths import get_path_for_dummy_dataset
from pytsbe.validation.validation import simple_train_test_split, in_sample_splitting


def get_dummy_dataset(dataset_name: str = 'dummy') -> TimeSeriesDatasets:
    """ Return simple dummy dataset for testing """
    dataset_processor = dataclass_by_name[dataset_name]
    dataset = dataset_processor.configure_dataset_from_path(dataset_name=dataset_name)
    return dataset


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


def test_train_test_split_univariate_ts():
    """ Check if train test splitting perform correctly for simple validation """
    forecast_horizon = 10
    dummy_dataset = get_dummy_dataset()
    ts_to_check = dummy_dataset.get_time_series_by_label('series_1')

    train_values, historical_values_for_test, test_values = simple_train_test_split(ts_to_check,
                                                                                    horizon=forecast_horizon)

    assert len(train_values) == len(historical_values_for_test) == 90
    assert len(test_values) == forecast_horizon
    assert test_values['value'].iloc[0] == 90


def test_in_sample_train_test_split_univariate_ts():
    """ Check if in sample splitting perform correctly """
    forecast_horizon = 20
    validation_blocks = 2

    dummy_dataset = get_dummy_dataset()
    ts_to_check = dummy_dataset.get_time_series_by_label('series_1')

    expected_train_length = 60
    for train_values, historical_values_for_test, test_values in in_sample_splitting(ts_to_check,
                                                                                     forecast_horizon,
                                                                                     validation_blocks):

        assert len(train_values) == len(historical_values_for_test) == expected_train_length
        assert test_values['value'].iloc[0] == expected_train_length
        expected_train_length += forecast_horizon


def test_train_test_split_multivariate_ts():
    """ Check simple train test split perform correctly for simple validation """
    forecast_horizon = 10
    dummy_dataset = get_dummy_dataset('multivariate_dummy')

    correct_target_cols = ['target_series_1', 'target_series_2', 'target_series_3']
    correct_first_test_elements = [90, 590, 890]
    for i, ts in enumerate(dummy_dataset.time_series):
        train_values, historical_values_for_test, test_values = simple_train_test_split(ts,
                                                                                        horizon=forecast_horizon)

        assert correct_target_cols[i] in list(train_values.columns)
        assert len(train_values) == 90
        assert test_values[correct_target_cols[i]].iloc[0] == correct_first_test_elements[i]


def test_in_sample_train_test_split_multivariate_ts():
    """ Check correctness of time series in-sample validation splitting for multivariate case """
    forecast_horizon = 10
    validation_blocks = 2

    correct_train_lens = [80, 90]
    first_time_in_test = [datetime.strptime('Jan 2 1998  8:00AM', '%b %d %Y %I:%M%p'),
                          datetime.strptime('Jan 2 1998  6:00PM', '%b %d %Y %I:%M%p')]
    dummy_dataset = get_dummy_dataset('multivariate_dummy')
    for ts in dummy_dataset.time_series:
        i = 0
        for train_values, historical_values_for_test, test_values in in_sample_splitting(ts,
                                                                                         forecast_horizon,
                                                                                         validation_blocks):
            # Perform checking
            assert correct_train_lens[i] == len(train_values) == len(historical_values_for_test)
            assert first_time_in_test[i] == test_values['datetime'].iloc[0]
            i += 1
