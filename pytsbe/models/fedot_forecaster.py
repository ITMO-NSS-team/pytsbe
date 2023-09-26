import pandas as pd
import numpy as np


try:
    from fedot.api.main import Fedot
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import TsForecastingParams, TaskTypesEnum, Task
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
        default_params = {'timeout': 5,
                          'n_jobs': 1}
        self.fit_params = {'predefined_model': params.get('predefined_model', None)}

        params.pop('predefined_moedel', 0)
        if params is not None:
            self.init_params = {**default_params, **params}
        else:
            self.init_params = default_params

        self.obtained_pipeline = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Train FEDOT framework (launch AutoML algorithm) """
        train_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=False)

        # Initialize model
        task_parameters = TsForecastingParams(forecast_length=forecast_horizon)
        self.model = Fedot(problem='ts_forecasting', task_params=task_parameters,
                           **self.init_params)

        self.obtained_pipeline = self.model.fit(features=train_data,
                                                **self.fit_params)

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        """ Create pipeline for multivariate time series forecasting """
        # Find target column
        train_data = {}
        for predictor_column in predictors_columns:
            train_data.update({str(predictor_column): np.array(historical_values[predictor_column])})

        task_parameters = TsForecastingParams(forecast_length=forecast_horizon)
        self.model = Fedot(problem='ts_forecasting', task_params=task_parameters,
                           **self.init_params)
        self.obtained_pipeline = self.model.fit(features=train_data,
                                                target=np.array(historical_values[target_column]),
                                                **self.fit_params)

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Use obtained pipeline to make predictions """
        historical_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=True)
        forecast = self.model.forecast(historical_data)

        result = ForecastResults(predictions=forecast, obtained_model=self.obtained_pipeline,
                                 additional_info={'fedot_api_object': self.model})
        return result

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        predict_input = {}
        for predictor_column in predictors_columns:
            predict_input.update({str(predictor_column): np.array(historical_values[predictor_column])})
        forecast = self.model.forecast(predict_input)

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
