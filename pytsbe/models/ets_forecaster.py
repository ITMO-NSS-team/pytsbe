import pandas as pd
import numpy as np

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

import logging
logging.raiseExceptions = False


class ETSForecaster(Forecaster):
    """
    Class for time series forecasting with exponential smoothing
    Source code:
    """

    def __init__(self, **params):
        super().__init__(**params)
        default_params = {'smoothing_level': 0.5, 'initialization_method': 'estimated'}
        if params is not None:
            self.params = {**default_params, **params}
        else:
            self.params = default_params

    def fit(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        pass

    def predict(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs) -> ForecastResults:
        # Get hyperparameters for current model
        alpha = self.params['smoothing_level']
        init_method = self.params['initialization_method']

        time_series = np.array((historical_values['value']))
        self.model = SimpleExpSmoothing(endog=time_series,
                                        initialization_method=init_method).fit(smoothing_level=alpha, optimized=False)

        predicted_ts = self.model.forecast(forecast_horizon)
        return ForecastResults(predictions=predicted_ts)
