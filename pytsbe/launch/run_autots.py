import time
import timeit
from copy import copy

import pandas as pd
import numpy as np

import warnings

from typing import List, Optional

from autots import AutoTS, model_forecast

from pytsbe.data import TimeSeriesDatasets
from pytsbe.launch.ts_launcher import TsRun, DEFAULT_FAILURES_THRESHOLD

warnings.filterwarnings('ignore')

from pylab import rcParams
rcParams['figure.figsize'] = 11, 4


class AutoTSTsRun(TsRun):
    """ Class for time series forecasting with AutoTS framework. Allows to make
    forecasts for selected time series, save forecasts and additional info:
    composing history, pictures of obtained pipelines and serialised models
    """

    def __init__(self, val_set: TimeSeriesDatasets, working_dir: str,
                 params: dict = None, launches: int = 1):
        super().__init__(val_set, working_dir, launches)
        default_params = {'frequency': 'infer', 'prediction_interval': 0.9,
                          'ensemble': 'all', 'model_list': 'superfast',
                          'max_generations': 10, 'num_validations': 2}
        if params is not None:
            self.params = {**default_params, **params}
        else:
            self.params = default_params

    def perform_validation(self, horizons: List[int], validation_blocks: Optional[int] = None):
        """ Start validation on provided datasets

        :param horizons: forecast horizons to process
        :param validation_blocks: validation blocks for in-sample forecasting
        """
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
                    predicted_values, model_name, train_dataset, time_launch = self.perform_experiment(failures,
                                                                                                       validation_blocks,
                                                                                                       len_forecast,
                                                                                                       dataset)
                    # Save predictions into csv file
                    self._save_forecast(label, train_dataset, dataset, predicted_values, len_forecast)

                    # Update info about runtime
                    times.append(time_launch)
                    launch_id.append(''.join((label, '_', str(len_forecast))))
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

        copied_dataset = copy(dataset)
        try:
            start = timeit.default_timer()

            if validation_blocks is None:
                # Prepare train part of dataset
                train_dataset = copied_dataset.head(len(dataset) - len_forecast)
                predicted_values, model_name = self._make_forecast(train_dataset, len_forecast)
            else:
                # Perform in-sample forecasting
                predicted_values, model_name = self._make_in_sample_forecast(copied_dataset, len_forecast,
                                                                             validation_blocks)

                # Clip source dataframe
                horizon = len_forecast * validation_blocks
                train_dataset = dataset.head(len(dataset) - horizon)

            time_launch = timeit.default_timer() - start
            return predicted_values, model_name, train_dataset, time_launch

        except Exception as ex:
            print(f'Restart launch for horizon {len_forecast} due to exception {ex}')
            time.sleep(15)
            failures += 1
            predicted_values, model_name, train_dataset, time_launch =\
                self.perform_experiment(failures, validation_blocks, len_forecast, dataset)

            return predicted_values, model_name, train_dataset, time_launch

    def _make_forecast(self, df, len_forecast: int):
        """
        Function for making time series forecasting with AutoTS library

        :param df: dataframe to process
        :param len_forecast: forecast length
        :return predicted_values: forecast of the model
        """
        time_series_label = 'value'
        model = self._configure_autots_model(len_forecast)

        model = model.fit(df, date_col='datetime', value_col=time_series_label)

        prediction = model.predict()
        # point forecasts dataframe
        forecasts_df = prediction.forecast

        predicted_values = np.array(forecasts_df[time_series_label])
        return predicted_values, str(model.best_model_name)

    def _make_in_sample_forecast(self, df, len_forecast: int, validation_blocks: int):
        """ Perform in sample forecasting for AutoTS model """
        time_series_label = 'value'
        model = self._configure_autots_model(len_forecast)

        horizon = validation_blocks * len_forecast
        df_train = df.head(len(df) - horizon)

        # Search for best model
        model = model.fit(df_train, date_col='datetime', value_col=time_series_label)

        all_forecasts = []
        for i in np.arange(validation_blocks, 0, -1):
            current_horizon = i * len_forecast
            df_train = df.head(len(df) - current_horizon)
            df_train['datetime'] = pd.to_datetime(df_train['datetime'])
            df_train = df_train.set_index('datetime')

            prediction = model_forecast(model_name=model.best_model_name,
                                        model_param_dict=model.best_model_params,
                                        model_transform_dict=model.best_model_transformation_params,
                                        df_train=df_train, forecast_length=len_forecast,
                                        frequency=self.params.get('frequency'),
                                        prediction_interval=self.params.get('prediction_interval'),
                                        no_negatives=False)
            forecasts_df = prediction.forecast
            predicted_values = list(forecasts_df[time_series_label])
            all_forecasts.extend(predicted_values)

        all_forecasts = np.array(all_forecasts)
        return all_forecasts, model.best_model_name

    def _configure_autots_model(self, len_forecast):
        frequency = self.params.get('frequency')
        prediction_interval = self.params.get('prediction_interval')
        ensemble = self.params.get('ensemble')
        model_list = self.params.get('model_list')
        max_generations = self.params.get('max_generations')
        num_validations = self.params.get('num_validations')

        model = AutoTS(forecast_length=len_forecast,
                       frequency=frequency,
                       prediction_interval=prediction_interval,
                       ensemble=ensemble,
                       model_list=model_list,
                       max_generations=max_generations,
                       num_validations=num_validations,
                       validation_method="backwards")
        return model
