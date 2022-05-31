import pytest

from test.test_report import get_results_for_testing
from pytsbe.main import TimeSeriesLauncher


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
