import pandas as pd
import numpy as np
from fedot.api.main import Fedot

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.repository.dataset_types import DataTypesEnum

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster


class FedotForecaster(Forecaster):
    """ Class for time series forecasting with FEDOT framework """

    def __init__(self, **params):
        super().__init__(**params)
        self.obtained_pipeline = None
        self.timeout = 5
        if 'timeout' in params:
            # Set new value for timeout
            self.timeout = params['timeout']

        self.preset = 'ts'
        if 'preset' in params:
            # Set new value for timeout
            self.preset = params['preset']

    def fit(self, historical_values: pd.DataFrame, forecast_horizon: int):
        """ Train FEDOT framework (launch AutoML algorithm) """
        train_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=False)

        # Initialize model
        task_parameters = TsForecastingParams(forecast_length=forecast_horizon)
        self.model = Fedot(problem='ts_forecasting', task_params=task_parameters,
                           timeout=self.timeout, preset=self.preset)

        # TODO change predefined_model after all experiments
        self.obtained_pipeline = self.model.fit(features=train_data)

    def predict(self, historical_values: pd.DataFrame, forecast_horizon: int) -> ForecastResults:
        """ Use obtained pipeline to make predictions """
        historical_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=True)
        forecast = self.model.predict(historical_data)

        # Generate new dataframe
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
