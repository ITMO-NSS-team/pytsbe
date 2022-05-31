import pandas as pd
import numpy as np

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging
logging.raiseExceptions = False


class NaiveRepeatLastValueForecaster(Forecaster):
    """
    Class for naive time series forecasting. Repeat last observation n times
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.last_observation = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        pass

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, exogenous_columns: list, **kwargs):
        pass

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                              **kwargs) -> ForecastResults:
        time_series = np.array(historical_values['value'])
        self.last_observation = time_series[-1]

        predicted_ts = np.full(forecast_horizon, self.last_observation)
        return ForecastResults(predictions=predicted_ts)

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, exogenous_columns: list, **kwargs):
        raise NotImplementedError('Naive forecaster does not support predict for multivariate time series forecasting')
