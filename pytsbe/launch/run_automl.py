import os
import time
import timeit
from abc import abstractmethod

import numpy as np

import warnings

from typing import List, Optional

from fedot.core.repository.operation_types_repository import OperationTypesRepository
from sklearn.metrics import mean_absolute_error

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast

from pytsbe.data import TimeSeriesDatasets
from pytsbe.launch.ts_launcher import TsRun, prepare_input_ts_data, DEFAULT_FAILURES_THRESHOLD

warnings.filterwarnings('ignore')

from pylab import rcParams
rcParams['figure.figsize'] = 11, 4

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.repository.dataset_types import DataTypesEnum


class AutoMLfromFedotTsRun(TsRun):
    """ Base class for solving time series forecasting task with different AutoML frameworks
    (H2O or TPOT). Such a frameworks are using as operations in the nodes.
    """

    def __init__(self, val_set: TimeSeriesDatasets, working_dir: str,
                 params: dict = None, launches: int = 1):
        super().__init__(val_set, working_dir, launches)
        self.time_part_for_tuning = None
        self.params_for_automl = None

    @abstractmethod
    def configure_automl_pipeline(self, len_forecast, window_size):
        """ Create pipeline with lagged transformation for AutoML library """
        raise NotImplementedError()

    @staticmethod
    def __configure_linear_pipeline(len_forecast):
        """ Create pipeline with lagged transformation and ridge regression """
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_forecast))

        lagged_node = PrimaryNode('lagged')
        ridge_node = SecondaryNode('ridge', nodes_from=[lagged_node])
        pipeline = Pipeline(ridge_node)
        return pipeline, task

    def perform_validation(self, horizons: List[int], validation_blocks: Optional[int] = None):
        """ Start validation on provided datasets

        :param horizons: forecast horizons to process
        :param validation_blocks: validation blocks for in-sample forecasting
        """
        with OperationTypesRepository.init_automl_repository() as _:
            for launch_number in range(0, self.launches):
                print(f'LAUNCH {launch_number}')
                launch_name = ''.join(('launch_', str(launch_number)))
                self.current_forecasts_path = self.forecasts_paths[launch_number]

                times = []
                launch_id = []
                models = []
                for dataset, label in zip(self.val_set.datasets, self.val_set.labels):
                    print(f'Processing dataset with label {label}')

                    # For every forecast horizon
                    for len_forecast in horizons:
                        failures = 0
                        predicted_values, pipeline, train_dataset, time_launch = \
                            self.perform_experiment(failures, validation_blocks, len_forecast, dataset)

                        # Save predictions into csv file
                        self._save_forecast(label, train_dataset, dataset, predicted_values, len_forecast)

                        # Update info about runtime
                        times.append(time_launch)
                        launch_id.append(''.join((label, '_', str(len_forecast))))

                        # Get information about lagged transformation
                        model_name = ''.join(('lagged_info', str(pipeline.nodes[-1].custom_params)))
                        models.append(model_name)

                self.save_report_csv(times, launch_id, launch_name, models)

    def perform_experiment(self, failures, validation_blocks, len_forecast, dataset):
        """ Obtain forecasts for desired horizon

        :param failures: number of run failures for considering forecast length
        :param validation_blocks: number of validation blocks
        :param len_forecast: forecast horizon
        :param dataset: dataframe with time series
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
                predicted_values, pipeline = self._make_forecast(train_dataset, len_forecast)
            else:
                # Perform in-sample validation
                predicted_values, pipeline = self._make_in_sample_forecast(dataset, len_forecast, validation_blocks)
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
                self.perform_experiment(failures, validation_blocks, len_forecast, dataset)

            return predicted_values, pipeline, train_dataset, time_launch

    def _make_forecast(self, df, len_forecast: int):
        """
        Method for making time series forecasting with TPOT framework

        :param df: dataframe to process
        :param len_forecast: forecast length

        :return predicted_values: forecast
        :return pipeline: obtained pipeline
        """
        # Prepare linear pipeline
        linear_pipeline, task = self.__configure_linear_pipeline(len_forecast)

        # Prepare data and tune linear pipeline
        input_data, predict_input = prepare_input_ts_data(df, task, len_forecast)
        tpot_pipeline = self._substitute_automl_training(linear_pipeline, input_data, len_forecast)

        predicted_output = tpot_pipeline.predict(predict_input)
        predicted_values = np.array(predicted_output.predict)

        return predicted_values, tpot_pipeline

    def _make_in_sample_forecast(self, df, len_forecast: int, validation_blocks: int):
        """ In-sample forecasting performed """
        time_series_label = 'value'
        linear_pipeline, task = self.__configure_linear_pipeline(len_forecast)

        # Define horizon
        horizon = len_forecast * validation_blocks
        time_series = np.array(df[time_series_label])
        train_part = time_series[:-horizon]

        input_data = InputData(idx=range(0, len(train_part)),
                               features=train_part, target=train_part,
                               task=task, data_type=DataTypesEnum.ts)
        validation_input = InputData(idx=range(0, len(time_series)),
                                     features=time_series, target=time_series,
                                     task=task, data_type=DataTypesEnum.ts)

        tpot_pipeline = self._substitute_automl_training(linear_pipeline, input_data, len_forecast)
        forecast = in_sample_ts_forecast(pipeline=tpot_pipeline,
                                         input_data=validation_input,
                                         horizon=horizon)

        return forecast, tpot_pipeline

    def _substitute_automl_training(self, linear_pipeline: Pipeline, input_data: InputData,
                                    len_forecast: int) -> Pipeline:
        """
        Using linear pipeline, tune lagged operation and then replace linear model (ridge)
        with AutoML as operation in the node

        :param linear_pipeline: pipeline with structure lagged -> ridge regression
        :param input_data: InputData for train
        :param len_forecast: forecast horizon
        """
        pipeline = linear_pipeline.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                                       input_data=input_data,
                                                       iterations=500,
                                                       timeout=self.time_part_for_tuning)

        lagged_params = pipeline.nodes[-1].custom_params
        window_size = lagged_params['window_size']
        # Get new pipeline with AutoML as operation
        tpot_pipeline, _ = self.configure_automl_pipeline(len_forecast, window_size)
        tpot_pipeline.fit(input_data)

        return tpot_pipeline


class TPOTTsRun(AutoMLfromFedotTsRun):
    """ Class for time series forecasting with TPOT framework. Allows to make
    forecasts for selected time series, save forecasts and additional info

    Important! TPOT using as Model in FEDOT framework due to it is unable to use
    TPOT directly for time series forecasting
    """

    def __init__(self, val_set: TimeSeriesDatasets, working_dir: str,
                 params: dict = None, launches: int = 1):
        super().__init__(val_set=val_set, working_dir=working_dir,
                         launches=launches)
        timeout = params['timeout']
        self.time_part_for_tuning = 0.3 * timeout
        self.time_part_for_tpot = round(0.7 * timeout)

        default_params = {'generations': 3, 'population_size': 2}
        self.params_for_automl = {**default_params, **params}
        self.params_for_automl.update({'timeout': self.time_part_for_tpot})

    def configure_automl_pipeline(self, len_forecast, window_size):
        """ Create pipeline with lagged transformation for TPOT framework """
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_forecast))

        lagged_node = PrimaryNode('lagged')
        lagged_node.custom_params = {'window_size': window_size}

        # Define custom params for TPOT
        tpot_node = SecondaryNode('tpot_regr', nodes_from=[lagged_node])
        tpot_node.custom_params = self.params_for_automl

        pipeline = Pipeline(tpot_node)
        return pipeline, task


class H2OTsRun(AutoMLfromFedotTsRun):
    """ Class for time series forecasting with H2O framework. Allows to make
    forecasts for selected time series, save forecasts and additional info

    Important! H2O using as Model in FEDOT framework due to it is unable to use
    H2O directly for time series forecasting
    """

    def __init__(self, val_set: TimeSeriesDatasets, working_dir: str,
                 params: dict = None, launches: int = 1):
        super().__init__(val_set=val_set, working_dir=working_dir,
                         launches=launches)
        timeout = params['timeout']
        self.time_part_for_tuning = 0.3 * timeout
        self.time_part_for_h2o = round(0.7 * timeout)

        default_params = {"timeout": 20, "seed": 42, "max_models": 3}
        self.params_for_automl = {**default_params, **params}
        self.params_for_automl.update({'timeout': self.time_part_for_h2o})

    def configure_automl_pipeline(self, len_forecast, window_size):
        """ Create pipeline with lagged transformation for TPOT framework """
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_forecast))

        lagged_node = PrimaryNode('lagged')
        lagged_node.custom_params = {'window_size': window_size}

        # Define custom params for H2O
        h2o_node = SecondaryNode('h2o_regr', nodes_from=[lagged_node])
        h2o_node.custom_params = self.params_for_automl

        pipeline = Pipeline(h2o_node)
        return pipeline, task
