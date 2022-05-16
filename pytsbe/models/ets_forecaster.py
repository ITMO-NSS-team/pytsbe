import pandas as pd
import numpy as np

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster
from statsmodels.tsa.api import ExponentialSmoothing

import logging
logging.raiseExceptions = False


class ETSForecaster(Forecaster):
    """
    Class for time series forecasting with exponential smoothing
    Source code: https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
    """

    def __init__(self, **params):
        super().__init__(**params)
        default_params = {}
        if params is not None:
            self.params = {**default_params, **params}
        else:
            self.params = default_params

    def fit(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        self.model = ExponentialSmoothing()
        self.fit()

    def predict(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs) -> ForecastResults:
        """ Use obtained model to make predictions """
        # Update model weights
        self.model.update(np.array(historical_values['value']))
        predicted_ts = self.model.predict(forecast_horizon,
                                          return_conf_int=False)
        return ForecastResults(predictions=predicted_ts)
