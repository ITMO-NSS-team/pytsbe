import pandas as pd
import numpy as np

try:
    from autots import AutoTS, model_forecast
except ImportError:
    print('Does not found AutoTS library. Continue...')

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging
logging.raiseExceptions = False


class AutoTSForecaster(Forecaster):
    """
    Class for time series forecasting with AutoTS library
    Source code: https://github.com/winedarksea/AutoTS
    """

    def __init__(self, **params):
        super().__init__(**params)
        default_params = {'frequency': 'infer', 'prediction_interval': 0.9,
                          'ensemble': 'all', 'model_list': 'superfast',
                          'max_generations': 10, 'num_validations': 2}
        if params is not None:
            self.params = {**default_params, **params}
        else:
            self.params = default_params

    def fit(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        self._configure_autots_model(forecast_horizon)
        self.model.fit(historical_values, date_col='datetime', value_col='value')

    def predict(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs) -> ForecastResults:
        """ Use obtained model to make predictions """
        forecasts_df = model_forecast(model_name=self.model.best_model_name,
                                      model_param_dict=self.model.best_model_params,
                                      model_transform_dict=self.model.best_model_transformation_params,
                                      df_train=historical_values.set_index('datetime'),
                                      forecast_length=forecast_horizon,
                                      frequency=self.params.get('frequency'),
                                      prediction_interval=self.params.get('prediction_interval'),
                                      no_negatives=False)

        forecast = np.ravel(np.array(forecasts_df.forecast['value']))
        # For now save only forecasts - possible to extend
        result = ForecastResults(predictions=forecast)
        return result

    def _configure_autots_model(self, len_forecast):
        frequency = self.params.get('frequency')
        prediction_interval = self.params.get('prediction_interval')
        ensemble = self.params.get('ensemble')
        model_list = self.params.get('model_list')
        max_generations = self.params.get('max_generations')
        num_validations = self.params.get('num_validations')

        self.model = AutoTS(forecast_length=len_forecast,
                            frequency=frequency,
                            prediction_interval=prediction_interval,
                            ensemble=ensemble,
                            model_list=model_list,
                            max_generations=max_generations,
                            num_validations=num_validations,
                            validation_method="backwards")
