import shutil

import pytest
import os
import pandas as pd

from pytsbe.paths import get_test_path
from pytsbe.report.visualisation import Visualizer
from test.test_report import get_results_for_testing
from pytsbe.main import TimeSeriesLauncher


@pytest.fixture(scope='session', autouse=True)
def delete_files_after_tests(request):
    paths = ['test_launch', 'test_multivariate_launch', 'path_for_plots']
    delete_files = create_func_delete_files(paths)
    delete_files()
    request.addfinalizer(delete_files)


def create_func_delete_files(paths):
    def wrapper():
        for path in paths:
            path = os.path.join(get_test_path(), path)
            if path is not None and os.path.isdir(path):
                shutil.rmtree(path)
    return wrapper


def test_launch_with_non_matching_configuration():
    """
    The attempt to run the algorithm must be terminated because a different
    configuration of the experiment is stored in the existing directory.
    """
    dir_with_results = get_results_for_testing()

    experimenter = TimeSeriesLauncher(working_dir=dir_with_results,
                                      datasets=['FRED', 'SMART'],
                                      launches=2)
    with pytest.raises(ValueError) as exc:
        experimenter.perform_experiment(libraries_to_compare=['FEDOT', 'repeat_last'],
                                        horizons=[10, 50],
                                        libraries_params={'FEDOT': {'preset': 'ts', 'timeout': 2}},
                                        validation_blocks=2,
                                        clip_border=500)
    assert str(exc.value) == 'New and old experiment configurations differ! ' \
                             'Roll back configuration or create new directory for experiments.'


def test_univariate_algorithm_launch():
    """ Check if algorithm can be launched correctly with desired time series and libraries """
    path_with_results = os.path.join(get_test_path(), 'test_launch')
    experimenter = TimeSeriesLauncher(working_dir=path_with_results,
                                      datasets=['FRED'],
                                      launches=1)
    experimenter.perform_experiment(libraries_to_compare=['repeat_last'],
                                    horizons=[3, 5],
                                    validation_blocks=2,
                                    clip_border=500)

    expected_fred_path = os.path.join(path_with_results, 'FRED')
    expected_library_path = os.path.join(expected_fred_path, 'launch_0', 'repeat_last')
    forecast_dataframe = pd.read_csv(os.path.join(expected_library_path, 'EXCAUS_5_forecast_vs_actual.csv'))

    assert os.path.isdir(expected_fred_path)
    assert len(os.listdir(expected_library_path)) == 48
    # Full validation length = forecast length * number of validation blocks
    assert len(forecast_dataframe) == 10


def test_multivariate_algorithm_launch():
    """ Launch multivariate time series forecasting with naive forecaster """
    path_with_results = os.path.join(get_test_path(), 'test_multivariate_launch')
    experimenter = TimeSeriesLauncher(working_dir=path_with_results,
                                      datasets=['SSH'], launches=1)
    experimenter.perform_experiment(libraries_to_compare=['average'], horizons=[10],
                                    validation_blocks=2, clip_border=400)

    expected_dataset_path = os.path.join(path_with_results, 'SSH')
    expected_library_path = os.path.join(expected_dataset_path, 'launch_0', 'average')
    forecast_dataframe = pd.read_csv(os.path.join(expected_library_path, '0_10_forecast_vs_actual.csv'))

    assert os.path.isdir(expected_dataset_path)
    assert len(os.listdir(expected_library_path)) == 50
    assert len(forecast_dataframe) == 20


def test_report_execution_time_plots():
    """ Check if folder with plots create correctly. Also check number of plots """
    folder_for_plots = os.path.join(get_test_path(), 'path_for_plots')
    plots_creator = Visualizer(working_dir=get_results_for_testing(),
                               folder_for_plots=folder_for_plots)
    plots_creator.execution_time_comparison()

    assert len(os.listdir(folder_for_plots)) == 2
