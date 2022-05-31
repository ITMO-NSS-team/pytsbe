import pandas as pd
import numpy as np

try:
    import pmdarima as pm
except ImportError:
    print('Does not found pmdarima library. Continue...')

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging
logging.raiseExceptions = False


class ARIMAForecaster(Forecaster):
    """
    Class for time series forecasting with pmdarima
    Source code: https://github.com/alkaline-ml/pmdarima
    """

    def __init__(self, **params):
        super().__init__(**params)
        default_params = {'start_p': 1, 'start_q': 1,
                          'd': 0, 'max_p': 6, 'max_q': 6,
                          'm': 24, 'suppress_warnings': True,
                          'stepwise': True, 'error_action': 'ignore',
                          'seasonal': True, 'trace': True, 'maxiter': 5}
        if params is not None:
            self.params = {**default_params, **params}
        else:
            self.params = default_params

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        time_series = np.array(historical_values['value'])
        self.model = pm.auto_arima(time_series, start_p=self.params['start_p'],
                                   start_q=self.params['start_q'], d=self.params['d'],
                                   max_p=self.params['max_p'], max_q=self.params['max_q'],
                                   m=self.params['m'], suppress_warnings=self.params['suppress_warnings'],
                                   stepwise=self.params['stepwise'], error_action=self.params['error_action'],
                                   seasonal=self.params['seasonal'], trace=self.params['trace'],
                                   maxiter=self.params['maxiter'])

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, exogenous_columns: list, **kwargs):
        raise NotImplementedError('pmdarima does not support fit for multivariate time series forecasting')

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                              **kwargs) -> ForecastResults:
        """ Use obtained model to make predictions """
        # Update model weights
        self.model.update(np.array(historical_values['value']))
        predicted_ts = self.model.predict(forecast_horizon,
                                          return_conf_int=False)
        return ForecastResults(predictions=predicted_ts)

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, exogenous_columns: list, **kwargs):
        raise NotImplementedError('pmdarima does not support predict for multivariate time series forecasting')
