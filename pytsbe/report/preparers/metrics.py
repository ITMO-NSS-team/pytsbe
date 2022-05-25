import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,\
    mean_squared_error

from pytsbe.report.preparers.utils import get_label_and_horizon
from pytsbe.report.walk import FolderWalker

import warnings
warnings.filterwarnings('ignore')


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """ Calculate symmetric mean absolute percentage error """
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    result = numerator / denominator
    result[np.isnan(result)] = 0.0
    return float(np.mean(100 * result))


metric_by_name = {'MAE': mean_absolute_error,
                  'MSE': mean_squared_error,
                  'RMSE': root_mean_squared_error,
                  'MAPE': mean_absolute_percentage_error,
                  'SMAPE': symmetric_mean_absolute_percentage_error}


def collect_metrics_table(walker: FolderWalker, metrics: list) -> pd.DataFrame:
    """
    Create a table with metrics for each case

    :param walker: folder walker which contain information about directory structure
    :param metrics: list with metrics for computing
    """

    table = []
    for case_id, saved_paths in walker.forecast_files.items():
        dataset, launch, library = case_id.split('|')

        # Load dictionary with results
        for file_path in saved_paths:
            df = pd.read_csv(file_path, parse_dates=['datetime'])

            # Get forecast horizon and time series name
            ts_label, forecast_horizon = get_label_and_horizon(file_path, '_forecast_vs_actual.csv')

            metrics_info = [dataset, launch, library, ts_label, forecast_horizon]
            for metric_name in metrics:
                evaluation_function = metric_by_name[metric_name]
                metric_value = evaluation_function(np.array(df['value']), np.array(df['predict']))
                metrics_info.append(metric_value)

            table.append(metrics_info)

    columns = ['Dataset', 'Launch', 'Library', 'Label', 'Horizon']
    columns.extend(metrics)
    table = pd.DataFrame(table, columns=columns)
    return table

