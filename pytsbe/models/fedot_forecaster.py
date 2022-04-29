import pandas as pd
import numpy as np

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.repository.dataset_types import DataTypesEnum

from pytsbe.models.forecast import Forecaster


class FedotForecaster(Forecaster):
    """ Class for time series forecasting with FEDOT framework """

    def __init__(self, **params):
        super().__init__(**params)

    def fit(self, historical_values: pd.DataFrame, forecast_horizon: int):
        """ Train FEDOT framework """
        train_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=False)



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
