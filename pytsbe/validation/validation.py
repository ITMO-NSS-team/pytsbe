import pandas as pd
import numpy as np
from typing import Optional, List, Union

from pytsbe.check import FailedLaunchChecker
from pytsbe.data.data import TimeSeriesDatasets
from pytsbe.exception import ExceptionHandler
from pytsbe.models.autots_forecaster import AutoTSForecaster
from pytsbe.models.fedot_forecaster import FedotForecaster
from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.pmdarima_forecaster import ARIMAForecaster
from pytsbe.timer import BenchmarkTimer

import warnings
warnings.filterwarnings('ignore')


class Validator:
    """ Class for validation on only one selected dataset (one dataset contains several time series)
    for the required forecast horizons

    Important: responsible for time series (from datasets) and horizons cycles
    """
    forecaster_by_name = {'FEDOT': FedotForecaster,
                          'AutoTS': AutoTSForecaster,
                          'pmdarima': ARIMAForecaster,
                          'TPOT': None,
                          'H2O': None}

    def __init__(self, dataset_name: str, launch_number: int, library_name: str,
                 library_parameters: dict, library_serializer):
        if library_name not in self.forecaster_by_name:
            raise NotImplementedError(f'Library {library_name} is not supported yet')

        self.library_name = library_name
        self.library_parameters = library_parameters
        self.library_serializer = library_serializer
        self.timer = BenchmarkTimer()
        self.launch_status_checker = FailedLaunchChecker(storage_paths=library_serializer.storage_paths,
                                                         dataset_name=dataset_name,
                                                         launch_number=launch_number,
                                                         library_name=library_name)

    def perform_experiments_on_dataset_and_horizons(self, dataset: TimeSeriesDatasets,
                                                    horizons: List[int],
                                                    validation_blocks: Optional[int] = None):
        """
        Launch experiment with desired dataset and time series forecasting algorithms.
        Check forecasting efficiency on all forecast horizons and all time series from
        the dataset.
        """
        for i, time_series in enumerate(dataset.time_series):
            ts_label = dataset.labels[i]

            # Initialise forecaster
            if self.library_parameters is not None:
                forecaster = self.forecaster_by_name[self.library_name](**self.library_parameters)
            else:
                forecaster = self.forecaster_by_name[self.library_name]()

            for horizon in horizons:
                if self.launch_status_checker.was_case_finished(ts_label, horizon):
                    # Already finish calculation - skip this case
                    continue

                exception_handler = ExceptionHandler(ts_label, horizon)
                with exception_handler.safe_process_launch():
                    # Prepare model for current forecast horizon
                    results = self._perform_experiment_on_single_ts(forecaster, time_series, horizon, validation_blocks)

                    # Save all the necessary results
                    self.library_serializer.save_information(ts_label, horizon, results)

    def _perform_experiment_on_single_ts(self, forecaster, time_series: pd.DataFrame,
                                         horizon: int, validation_blocks: Union[int, None]) -> ForecastResults:
        """ Launch time series forecasting algorithm on single time series for particular forecast horizon

        :param forecaster: object, which can produce forecast and use fit and predict methods
        :param time_series: table with datetime and values columns
        :param horizon: forecast horizon length
        :param validation_blocks: number of blocks for in-sample validation
        """
        if validation_blocks is None or validation_blocks == 1:
            # Simple experiment - predict on only one fold
            train_values, historical_values_for_test, test_values = simple_train_test_split(time_series, horizon)

            with self.timer.launch_fit():
                forecaster.fit(train_values, horizon)
            with self.timer.launch_predict():
                forecast_output = forecaster.predict(historical_values=historical_values_for_test,
                                                     forecast_horizon=horizon)
            # Update result with additional information
            forecast_output.true_values = test_values
            forecast_output.timeouts = {'fit_seconds': self.timer.fit_time, 'predict_seconds': self.timer.predict_time}
        else:
            # In-sample validation required
            forecast_output = self._launch_in_sample_validation(forecaster, time_series,
                                                                horizon, validation_blocks)

        self.timer.reset_timers()
        return forecast_output

    def _launch_in_sample_validation(self, forecaster, time_series: pd.DataFrame,
                                     horizon: int, validation_blocks: int):
        """ Launch in sample time series for current time series and forecasting horizon """
        validation_block_number = 0
        results = []
        for train_values, historical_values_for_test, test_values in in_sample_splitting(time_series,
                                                                                         horizon,
                                                                                         validation_blocks):
            if validation_block_number == 0:
                with self.timer.launch_fit():
                    # Perform training only for first launch
                    forecaster.fit(train_values, horizon)

                with self.timer.launch_predict():
                    # Check run time only for first validation block
                    forecast_output = forecaster.predict(
                        historical_values=historical_values_for_test,
                        forecast_horizon=horizon)
            else:
                # Make only forecast
                forecast_output = forecaster.predict(
                    historical_values=historical_values_for_test,
                    forecast_horizon=horizon)

            forecast_output.true_values = test_values
            forecast_output.timeouts = {'fit_seconds': self.timer.fit_time,
                                        'predict_seconds': self.timer.predict_time}
            results.append(forecast_output)
            validation_block_number += 1

        # Perform union of in-sample forecasting results
        forecast_output = ForecastResults.union(results)
        return forecast_output


def simple_train_test_split(time_series: pd.DataFrame, horizon: int):
    """ Perform simple train test split for time series to make forecast on only one fold """
    train_len = len(time_series) - horizon

    train_values = time_series.head(train_len)
    historical_values_for_test = train_values.copy()
    test_values = time_series.tail(horizon)
    return train_values, historical_values_for_test, test_values


def in_sample_splitting(time_series: pd.DataFrame, horizon: int, validation_blocks: int):
    """
    Prepare several splits of historical values for time series in-sample
    forecasting and return it iteratively
    """
    source_ts_len = len(time_series)

    for validation_block in np.arange(validation_blocks, 0, -1):
        # Create dataframes per each validation block
        remained_part_for_test = horizon * validation_block

        last_train_index = source_ts_len - remained_part_for_test

        train_values = time_series.head(last_train_index)
        historical_values_for_test = train_values.copy()
        test_values = time_series.iloc[last_train_index: last_train_index + horizon, :]

        yield train_values, historical_values_for_test, test_values
