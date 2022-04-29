from pytsbe.data.data import TimeSeriesDatasets
from pytsbe.paths import get_path_for_dummy_dataset


def test_long_format_data_loaded_correctly():
    """ Check if the dataset in long format are loaded into memory correctly """
    dummy_path = get_path_for_dummy_dataset()
    ts_dataset = TimeSeriesDatasets.setup_from_long_format(path=dummy_path)

    assert len(ts_dataset.time_series) == 3
    assert len(ts_dataset.time_series[0]) == 100
    assert ts_dataset.labels[0] == 'series_1'
