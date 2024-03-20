import copy
import numpy as np
import pandas as pd
import torch

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

try:
    from chronos import ChronosPipeline
except ImportError:
    print('Try installing Chronos via pip install '
          'git+https://github.com/amazon-science/chronos-forecasting.git')


class ChronosForecaster(Forecaster):
    def __init__(self, **params):
        super().__init__(**params)
        self.target = 'value'
        self.forecaster = self.__load_pretrained_pipeline(params.get('hf_model', 'amazon/chronos-t5-tiny'))

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        pass

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, target_column: str,
                            predictors_columns: list, **kwargs):
        raise NotImplementedError('Chronos does not support fit for multivariate time series forecasting')

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        forecast = self.forecaster.predict(
            torch.tensor(historical_values[self.target].values),
            prediction_length=forecast_horizon
        )
        return ForecastResults(predictions=np.median(forecast[0].numpy(), axis=0))

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, target_column: str,
                                predictors_columns: list, **kwargs):
        raise NotImplementedError('Chronos does not support predict for multivariate time series forecasting')

    @staticmethod
    def __load_pretrained_pipeline(hf_model: str) -> ChronosPipeline:
        return ChronosPipeline.from_pretrained(
            hf_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
