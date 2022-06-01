import os

from pytsbe.paths import get_project_path
from pytsbe.report.preparers.utils import get_label_and_horizon
from pytsbe.report.report import MetricsReport


def get_results_for_testing():
    project_path = get_project_path()
    return os.path.join(project_path, 'test', 'data', 'results')


def test_report_with_unequal_outputs():
    """
    Not all experiments were saved in the results folder (for FRED two
    cases were skipped. Full number of cases should be 24 (12 * 2 launches)).
    Therefore, the reporting algorithm needs to exclude incomplete
    cases from the comparison.
    """
    number_remained_cases_fred = 22
    dir_with_results = get_results_for_testing()

    metrics_processor = MetricsReport(working_dir=dir_with_results)

    assert len(metrics_processor.walker.forecast_files['FRED|1|repeat_last']) == number_remained_cases_fred
    assert len(metrics_processor.walker.forecast_files['FRED|0|average']) == number_remained_cases_fred


def test_define_horizon_and_label_correctly():
    """ Function should parse names of resulted files """
    file_name = 'traffic_volume_10_forecast_vs_actual.csv'
    file_path = os.path.join(get_results_for_testing(), 'FRED', 'launch_0', 'average', file_name)
    ts_label, forecast_horizon = get_label_and_horizon(file_path, '_forecast_vs_actual.csv')

    assert ts_label == 'traffic_volume'
    assert forecast_horizon == 10
