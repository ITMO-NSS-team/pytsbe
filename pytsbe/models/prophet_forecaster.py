import pandas as pd
import numpy as np

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

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        historical_values = historical_values.rename(columns={'datetime': 'ds', 'value': 'y'})

        self.model = Prophet()
        self.model.fit(historical_values)

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, exogenous_columns: list, **kwargs):
        raise NotImplementedError('Prophet does not support fit for multivariate time series forecasting')

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                              **kwargs) -> ForecastResults:
        """ Use fitted model to prepare forecast """
        future = kwargs['future_indices']
        future = pd.DataFrame({'ds': future})

        forecast_df = self.model.predict(future)
        return ForecastResults(predictions=np.array(forecast_df['yhat']))

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, exogenous_columns: list, **kwargs):
        raise NotImplementedError('Prophet does not support predict for multivariate time series forecasting')
