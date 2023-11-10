import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import NBEATS

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging

logging.raiseExceptions = False


class NBEATSForecaster(Forecaster):
    """
    Class for time series forecasting with NBEATS
    Source code: https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nbeats.NBeats.html
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.freq = params.get('freq', 'D')
        self.accelerator = params.get('accelerator', 'cpu')
        self.model = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        self.model = self._init_model(forecast_horizon)
        historical_values = self.prepare_dataset(historical_values)
        self.model.fit(historical_values, val_size=forecast_horizon)

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('NBEATS does not support fit for multivariate time series forecasting')

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                              **kwargs) -> ForecastResults:
        """ Use obtained model to make predictions """
        forecasts = self.model.predict()
        return ForecastResults(predictions=forecasts['NBEATS-median'].values)

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('NBEATS does not support predict for multivariate time series forecasting')

    def _init_model(self, forecast_horizon):
        model = NBEATS(h=forecast_horizon, input_size=24,
                       loss=DistributionLoss(distribution='Poisson', level=[80, 90]),
                       stack_types=['identity', 'trend', 'seasonality'],
                       max_steps=500,
                       val_check_steps=10,
                       early_stop_patience_steps=2,
                       accelerator=self.accelerator)
        fcst = NeuralForecast(
            models=[model],
            freq=self.freq
        )
        return fcst

    def prepare_dataset(self, historical_values):
        historical_values = historical_values.copy()
        historical_values['unique_id'] = '0'
        historical_values.columns = ['ds', 'y', 'unique_id']
        historical_values['ds'] = pd.to_datetime(historical_values['ds'])
        return historical_values
