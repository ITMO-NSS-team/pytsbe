import shutil

import numpy as np
import pandas as pd


try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except Exception:
    print('Does not found AutoGluon library. Continue...')

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging

logging.raiseExceptions = False


class AutoGluonForecaster(Forecaster):
    """
    Class for time series forecasting with Autogluon
    Source code: https://github.com/autogluon/autogluon
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.target = 'value'
        self.timeout = params.get('timeout', 60)
        self.presets = params.get('presets')
        self.model = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        self.model = self._init_model(forecast_horizon)
        historical_values = historical_values.copy()
        historical_values['idx'] = '0'
        train_data = TimeSeriesDataFrame.from_data_frame(
            historical_values,
            id_column="idx",
            timestamp_column="datetime"
        )

        self.model.fit(
            train_data,
            presets=self.presets,
            time_limit=self.timeout,
        )

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('AutoGluon does not support fit for multivariate time series forecasting')

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                              **kwargs) -> ForecastResults:
        """ Use obtained model to make predictions """
        # Update model weights
        historical_values = historical_values.copy()
        historical_values['idx'] = '0'
        train_data = TimeSeriesDataFrame.from_data_frame(
            historical_values,
            id_column="idx",
            timestamp_column="datetime"
        )
        predictions = self.model.predict(train_data)
        predictions.head()
        shutil.rmtree('AutogluonModels')
        return ForecastResults(predictions=predictions['mean'].values)

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('LAMA does not support predict for multivariate time series forecasting')

    def _init_model(self, forecast_horizon):
        return TimeSeriesPredictor(
            prediction_length=forecast_horizon,
            target="value",
            eval_metric="sMAPE",
        )
