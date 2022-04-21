import os
from abc import abstractmethod
import pandas as pd
import numpy as np

from typing import List, Optional, Callable
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

from pytsbe.data import TimeSeriesDatasets


DEFAULT_FAILURES_THRESHOLD = 10


class TsRun:
    """ Base class for time series forecasting """

    def __init__(self, val_set: TimeSeriesDatasets, working_dir: str, launches: int = 1):
        # Set of datasets
        self.val_set = val_set
        self.launches = launches

        # Define working directory
        self.working_dir = working_dir
        self._create_folder(self.working_dir)

        # For every launch create own folder
        self.launches_forecasts = []
        self.forecasts_paths = []
        for launch_number in range(0, launches):
            launch_name = ''.join(('launch_', str(launch_number)))
            forecasts_path = os.path.join(self.working_dir, launch_name, 'forecasts')

            self.launches_forecasts.append(forecasts_path)
            self.forecasts_paths.append(forecasts_path)

            self._create_folder(forecasts_path)

        self.current_forecasts_path = None

    @abstractmethod
    def perform_validation(self, horizons: List[int], validation_blocks: Optional[int] = None):
        """ Start validation on provided datasets for chosen datasets

        :param horizons: forecast horizons to process
        :param validation_blocks: validation blocks for in-sample forecasting
        """
        raise NotImplementedError()

    def _save_forecast(self, label: str, train_dataset: pd.DataFrame,
                       dataset: pd.DataFrame, predicted_values: np.array,
                       len_forecast: int):
        """
        Save forecast in csv file with columns 'value' and 'predicted'
        :param label: label for time series
        :param train_dataset: dataframe for train
        :param dataset: full size dataframe
        :param predicted_values: array with predicted values
        :param len_forecast: forecast horizon
        """
        train_values = np.ravel(np.array(train_dataset['value']))
        dataset['predicted'] = np.hstack([train_values, predicted_values])

        file_name = ''.join((label, '_', str(len_forecast), '.csv'))
        path_to_save = os.path.join(self.current_forecasts_path, file_name)
        dataset.to_csv(path_to_save, index=False)

    @staticmethod
    def display_metrics(additional_metric: Callable = None):
        """ Print metrics for validation set

        :param additional_metric: metric to calculate
        """
        raise NotImplementedError()

    @staticmethod
    def plot_forecast():
        """ Plot forecast """
        raise NotImplementedError()

    def save_report_csv(self, times: list, launch_id: list, launch_name: str,
                        models: list = None):
        if models is None:
            report_df = pd.DataFrame({'time_sec': times,
                                      'run_id': launch_id})
        else:
            report_df = pd.DataFrame({'time_sec': times,
                                      'run_id': launch_id,
                                      'model': models})
        report_path = os.path.join(self.working_dir, launch_name, 'report.csv')
        report_df.to_csv(report_path, index=False)

    @staticmethod
    def _create_folder(path):
        # Create new folder if it's not exists
        if os.path.isdir(path) is False:
            os.makedirs(path)


def prepare_input_ts_data(df, task, len_forecast):
    """ Return converted into InputData datasets for train and for prediction """
    time_series_label = 'value'
    input_data = InputData(idx=np.arange(0, len(df)),
                           features=np.array(df[time_series_label]),
                           target=np.array(df[time_series_label]),
                           task=task,
                           data_type=DataTypesEnum.ts)

    start_forecast = len(df)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=np.array(df[time_series_label]),
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return input_data, predict_input
