import pandas as pd

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster
try:
    from prophet import Prophet
except ImportError:
    print('Does not found prophet library. Continue...')

import logging
logging.raiseExceptions = False


class ProphetForecaster(Forecaster):
    """
    Class for time series forecasting with prophet library
    Source code: https://github.com/facebook/prophet
    """

    def __init__(self, **params):
        super().__init__(**params)
        default_params = {}
        if params is not None:
            self.params = {**default_params, **params}
        else:
            self.params = default_params

    def fit(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        historical_values = historical_values.rename(columns={'datetime': 'ds', 'value': 'y'})

        self.model = Prophet()
        self.model.fit(historical_values)

    def predict(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs) -> ForecastResults:
        """ Use fitted model to prepare forecast """
        future = self.model.make_future_dataframe(periods=forecast_horizon,
                                                  include_history=False)
        # TODO fix for in-sample forecasting
        raise NotImplementedError()
