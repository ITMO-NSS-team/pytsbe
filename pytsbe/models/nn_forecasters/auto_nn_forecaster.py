from abc import abstractmethod

import numpy as np
import pandas as pd
from neuralforecast.tsdataset import TimeSeriesDataset

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging

logging.raiseExceptions = False


class NNForecaster(Forecaster):
    """
    Abstract class for time series forecasting with NN
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.accelerator = params.get('accelerator', 'cpu')
        self.model = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        self.model = self._init_model(forecast_horizon)
        historical_values = self.prepare_dataset(historical_values)
        self.model.fit(historical_values)

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                              **kwargs) -> ForecastResults:
        """ Use obtained model to make predictions """
        historical_values = self.prepare_dataset(historical_values)
        forecasts = self.model.predict(historical_values)
        return ForecastResults(predictions=np.ravel(forecasts))

    @abstractmethod
    def _init_model(self, forecast_horizon):
        raise NotImplementedError()

    def prepare_dataset(self, historical_values):
        historical_values = historical_values.copy()
        historical_values['unique_id'] = '0'
        historical_values.columns = ['ds', 'y', 'unique_id']
        historical_values['ds'] = pd.to_datetime(historical_values['ds'])
        return TimeSeriesDataset.from_df(historical_values)[0]
