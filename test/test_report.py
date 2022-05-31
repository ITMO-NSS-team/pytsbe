import os

from pytsbe.paths import get_project_path
from pytsbe.report.report import MetricsReport


def get_results_for_testing():
    project_path = get_project_path()
    return os.path.join(project_path, 'test', 'data', 'results')


def test_report_with_unequal_outputs():
    """
    Not all experiments were saved in the results folder (for FRED two cases were skipped).
    Therefore, the reporting algorithm needs to exclude incomplete
    cases from the comparison.
    """
    dir_with_results = get_results_for_testing()

    metrics_processor = MetricsReport(working_dir=dir_with_results)
    metrics_table = metrics_processor.metric_table(metrics=['MAE', 'SMAPE'],
                                                   aggregation=['Library', 'Dataset'])
