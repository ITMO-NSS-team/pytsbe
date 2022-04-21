import os
import time
import timeit

import numpy as np

import warnings

from typing import List, Optional

from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast

from pytsbe.data import TimeSeriesDatasets
from pytsbe.launch.ts_launcher import TsRun, prepare_input_ts_data, DEFAULT_FAILURES_THRESHOLD

warnings.filterwarnings('ignore')

from pylab import rcParams
rcParams['figure.figsize'] = 11, 4

from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.repository.dataset_types import DataTypesEnum


class FedotTsRun(TsRun):
    """ Class for time series forecasting with FEDOT framework. Allows to make
    forecasts for selected time series, save forecasts and additional info:
    composing history, pictures of obtained pipelines and serialised models
    """

    def __init__(self, val_set: TimeSeriesDatasets, working_dir: str,
                 params: dict = None, launches: int = 1):
        super().__init__(val_set, working_dir, launches)

        # For every launch prepared own supplementary directories
        self.pictures_paths = []
        self.pipelines_paths = []
        self.history_paths = []
        for launch_number in range(0, launches):
            launch_name = ''.join(('launch_', str(launch_number)))
            pictures_path = os.path.join(self.working_dir, launch_name, 'pictures')
            pipelines_path = os.path.join(self.working_dir, launch_name, 'pipelines', '')

            history_path = os.path.join(self.working_dir, launch_name, 'history')
            # Convert into absolute path
            history_path = os.path.abspath(history_path)

            # Create folders
            for folder in [pictures_path, pipelines_path, history_path]:
                self._create_folder(folder)

            self.pictures_paths.append(pictures_path)
            self.pipelines_paths.append(pipelines_path)
            self.history_paths.append(history_path)

        self.fedot_params = self._define_fedot_params(params)

        self.current_pictures_path = None
        self.current_pipelines_path = None
        self.current_history_path = None

    def _define_fedot_params(self, fedot_params):
        default_params = {'composer_params': {'timeout': 1, 'preset': 'ts_tun'},
                          'preset': 'ts_tun', 'timeout': 1}
        if fedot_params is None:
            # Default parameters for FEDOT launch launch
            fedot_params = default_params
        else:
            fedot_params = {**default_params, **fedot_params}
        return fedot_params

    def _save_picture(self, label: str, len_forecast: int, pipeline: Pipeline):
        """
        Save picture of obtained pipeline structure
        :param label: label for time series
        :param len_forecast: forecast horizon
        :param pipeline: obtained pipeline
        """
        file_name = ''.join((label, '_', str(len_forecast), '.png'))
        path = os.path.join(self.current_pictures_path, file_name)
        pipeline.show(path)

    def _save_pipeline(self, label: str, len_forecast: int, pipeline: Pipeline):
        # Save pipeline
        pipeline.save(self.current_pipelines_path)
        folders = os.listdir(self.current_pipelines_path)

        for folder in folders:
            if 'PM' in folder or 'AM' in folder:
                # Folder need to be renamed
                old_name = os.path.join(self.current_pipelines_path, folder)
                new_name = os.path.join(self.current_pipelines_path, ''.join((label, '_', str(len_forecast))))
                os.rename(old_name, new_name)

    def perform_validation(self, horizons: List[int], validation_blocks: Optional[int] = None):
        """ Start validation on provided datasets

        :param horizons: forecast horizons to process
        :param validation_blocks: validation blocks for in-sample forecasting
        """
        for launch_number in range(0, self.launches):
            print(f'LAUNCH {launch_number}')
            launch_name = ''.join(('launch_', str(launch_number)))
            self.current_pictures_path = self.pictures_paths[launch_number]
            self.current_pipelines_path = self.pipelines_paths[launch_number]
            self.current_history_path = self.history_paths[launch_number]
            self.current_forecasts_path = self.forecasts_paths[launch_number]

            times = []
            run_id = []
            for dataset, label in zip(self.val_set.datasets, self.val_set.labels):
                print(f'Processing dataset with label {label}')

                # For every forecast horizon
                for len_forecast in horizons:
                    failures = 0
                    predicted_values, pipeline, train_dataset, time_launch = self.perform_experiment(failures,
                                                                                                     validation_blocks,
                                                                                                     len_forecast,
                                                                                                     dataset,
                                                                                                     label)

                    # Save predictions into csv file
                    self._save_forecast(label, train_dataset, dataset, predicted_values, len_forecast)
                    self._save_picture(label, len_forecast, pipeline)
                    self._save_pipeline(label, len_forecast, pipeline)

                    # Update info about runtime
                    times.append(time_launch)
                    run_id.append(''.join((label, '_', str(len_forecast))))

                self._check_log_paths()

            self.save_report_csv(times, run_id, launch_name)

    def perform_experiment(self, failures, validation_blocks, len_forecast, dataset, label):
        """ Obtain forecasts for desired horizon

        :param failures: number of run failures for considering forecast length
        :param validation_blocks: number of validation blocks
        :param len_forecast: forecast horizon
        :param dataset: dataframe with time series
        :param label: label for time series (name of column)
        """
        if failures > DEFAULT_FAILURES_THRESHOLD:
            # Count number of failed launches to avoid looping
            raise ValueError('Too many exceptions for one dataset')
        try:
            start = timeit.default_timer()

            if validation_blocks is None:
                # Prepare train part of dataset
                train_dataset = dataset.head(len(dataset) - len_forecast)
                # Run AutoML algorithm for last part prediction
                predicted_values, pipeline = self._make_forecast(train_dataset, len_forecast, label)
            else:
                # Perform in-sample validation
                predicted_values, pipeline = self._make_in_sample_forecast(dataset, len_forecast,
                                                                           validation_blocks, label)
                # Clip source dataframe
                horizon = len_forecast * validation_blocks
                train_dataset = dataset.head(len(dataset) - horizon)

            time_launch = timeit.default_timer() - start
            return predicted_values, pipeline, train_dataset, time_launch
        except Exception as ex:
            print(f'Restart launch for horizon {len_forecast} due to exception {ex}')
            time.sleep(15)
            failures += 1
            predicted_values, pipeline, train_dataset, time_launch = \
                self.perform_experiment(failures, validation_blocks, len_forecast, dataset, label)

            return predicted_values, pipeline, train_dataset, time_launch

    def _make_forecast(self, df, len_forecast: int, label: str):
        """
        Method for making time series forecasting with FEDOT framework

        :param df: dataframe to process
        :param len_forecast: forecast length
        :param label: label for time series

        :return predicted_values: forecast
        :return pipeline: obtained pipeline
        """
        # Prepare Fedot class for forecasting
        model, task = self.__configure_fedot(label, len_forecast)

        # Prepare data
        input_data, predict_input = prepare_input_ts_data(df, task, len_forecast)
        pipeline = model.fit(features=input_data)
        predicted_values = model.predict(predict_input)

        return predicted_values, pipeline

    def _make_in_sample_forecast(self, df, len_forecast: int, validation_blocks: int,  label: str):
        """ In-sample forecasting performed """
        time_series_label = 'value'
        model, task = self.__configure_fedot(label, len_forecast)

        # Define horizon
        horizon = len_forecast * validation_blocks

        time_series = np.array(df[time_series_label])
        train_part = time_series[:-horizon]

        # InputData for train
        input_data = InputData(idx=np.arange(0, len(train_part)),
                               features=train_part, target=train_part,
                               task=task, data_type=DataTypesEnum.ts)

        # InputData for validation
        validation_input = InputData(idx=np.arange(0, len(time_series)),
                                     features=time_series, target=time_series,
                                     task=task, data_type=DataTypesEnum.ts)

        pipeline = model.fit(features=input_data)
        forecast = in_sample_ts_forecast(pipeline=pipeline,
                                         input_data=validation_input,
                                         horizon=horizon)

        return forecast, pipeline

    def _check_log_paths(self):
        for save_dir in [self.current_forecasts_path, self.current_pipelines_path]:
            files = os.listdir(save_dir)

            # Directories for storing results must be not empty
            if len(files) == 0:
                raise ValueError(f'Directory {save_dir} is empty.')

    def __configure_fedot(self, label, len_forecast):
        # Define parameters
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_forecast))

        # Init model for the time series forecasting
        composer_params = self.fedot_params['composer_params']
        history_path_to_save = os.path.join(self.current_history_path, ''.join((label, '_', str(len_forecast))))
        self._create_folder(history_path_to_save)

        composer_params = {**composer_params, **{'history_folder': history_path_to_save}}
        model = Fedot(problem='ts_forecasting', task_params=task.task_params,
                      composer_params=composer_params,
                      preset=self.fedot_params['preset'],
                      timeout=self.fedot_params['timeout'])

        return model, task
