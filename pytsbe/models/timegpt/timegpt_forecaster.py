import copy
import pandas as pd

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

try:
    from nixtlats import TimeGPT
except ImportError:
    print('Does not found nixtlats. Continue...')


class TimeGPTForecaster(Forecaster):
    """
    Class for time series forecasting with Nixtla TimeGPT
    Source code: https://github.com/Nixtla/nixtla
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.target = 'value'
        self.timegpt = TimeGPT(token=params.get('token'))
        if not self.timegpt.validate_token():
            raise Exception('Provide a valid TimeGPT token in configuration')

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        pass

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, target_column: str,
                            predictors_columns: list, **kwargs):
        pass

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        df = copy.deepcopy(historical_values)
        forecast = self.timegpt.forecast(
            df=df,
            h=forecast_horizon,
            time_col='datetime',
            target_col='value',
            freq='MS'
        )
        return ForecastResults(predictions=forecast['TimeGPT'].tolist())

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, target_column: str,
                                predictors_columns: list, **kwargs):
        raise NotImplementedError()
