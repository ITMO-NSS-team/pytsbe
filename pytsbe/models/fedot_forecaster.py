import pandas as pd
import numpy as np

try:
    from fedot.api.main import Fedot
    from fedot.core.data.data import InputData
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
    from fedot.core.pipelines.pipeline import Pipeline
except ImportError:
    print('Does not found FEDOT library. Continue...')

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging
logging.raiseExceptions = False


class FedotForecaster(Forecaster):
    """
    Class for time series forecasting with FEDOT framework
    Source code:
    """

    def __init__(self, **params):
        super().__init__(**params)
        # TODO refactor with dictionaries
        self.obtained_pipeline = None
        self.timeout = 5
        if 'timeout' in params:
            # Set new value for timeout
            self.timeout = params['timeout']

        self.preset = 'ts'
        if 'preset' in params:
            # Set new preset
            self.preset = params['preset']

        self.predefined_model = None
        if 'predefined_model' in params:
            # Set new preset
            self.predefined_model = params['predefined_model']

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Train FEDOT framework (launch AutoML algorithm) """
        train_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=False)

        # Initialize model
        task_parameters = TsForecastingParams(forecast_length=forecast_horizon)
        self.model = Fedot(problem='ts_forecasting', task_params=task_parameters,
                           timeout=self.timeout, preset=self.preset)

        self.obtained_pipeline = self.model.fit(features=train_data,
                                                predefined_model=self.predefined_model)

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        """ Create pipeline for multivariate time series forecasting """
        # Find target column
        train_data = {}
        for predictor_column in predictors_columns:
            train_data.update({str(predictor_column): np.array(historical_values[predictor_column])})

        task_parameters = TsForecastingParams(forecast_length=forecast_horizon)
        self.model = Fedot(problem='ts_forecasting', task_params=task_parameters,
                           timeout=self.timeout, preset=self.preset)
        self.obtained_pipeline = self.model.fit(features=train_data,
                                                target=np.array(historical_values[target_column]),
                                                predefined_model=self.predefined_model)

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Use obtained pipeline to make predictions """
        historical_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=True)
        forecast = self.model.predict(historical_data)

        result = ForecastResults(predictions=forecast, obtained_model=self.obtained_pipeline,
                                 additional_info={'fedot_api_object': self.model})
        return result

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        predict_input = {}
        for predictor_column in predictors_columns:
            predict_input.update({str(predictor_column): np.array(historical_values[predictor_column])})
        forecast = self.model.predict(predict_input)

        result = ForecastResults(predictions=forecast, obtained_model=self.obtained_pipeline,
                                 additional_info={'fedot_api_object': self.model})
        return result


def prepare_input_ts_data(historical_values: pd.DataFrame, forecast_horizon: int,
                          is_for_forecast: bool = False):
    """ Return converted into InputData datasets for train and for prediction """
    time_series_label = 'value'
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_horizon))

    if is_for_forecast:
        start_forecast = len(historical_values)
        end_forecast = start_forecast + forecast_horizon

        input_data = InputData(idx=np.arange(start_forecast, end_forecast),
                               features=np.array(historical_values[time_series_label]),
                               target=np.array(historical_values[time_series_label]),
                               task=task, data_type=DataTypesEnum.ts)
    else:
        input_data = InputData(idx=np.arange(0, len(historical_values)),
                               features=np.array(historical_values[time_series_label]),
                               target=np.array(historical_values[time_series_label]),
                               task=task, data_type=DataTypesEnum.ts)

    return input_data


def get_simple_pipeline():
    """ Create pipeline with lagged transformation and decision tree regression """
    lagged_node = PrimaryNode('lagged')
    dtreg_node = SecondaryNode('dtreg', nodes_from=[lagged_node])
    pipeline = Pipeline(dtreg_node)
    return pipeline
