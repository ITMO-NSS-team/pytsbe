import pandas as pd
import numpy as np

try:
    from lightautoml.tasks import Task as LAMAtask
    from lightautoml.addons.autots.base import AutoTS
except ImportError:
    print('Does not found LAMA library. Continue...')


from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging
logging.raiseExceptions = False


class LAMAForecaster(Forecaster):
    """
    Class for time series forecasting with LightAutoML
    Source code: https://github.com/sb-ai-lab/LightAutoML
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.target = 'value'
        self.task = LAMAtask('multi:reg', greater_is_better=False, metric="mae", loss="mae")
        self.model = None
    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        historical_values['datetime'] = pd.to_datetime(historical_values['datetime'])
        train = historical_values.reset_index()[['datetime', 'value']]
        TARGET = 'value'
        seq_params = {"seq0": {"case": "next_values",
                               "params": {"n_target": forecast_horizon,
                                          "history": np.maximum(7, forecast_horizon),
                                          "step": 1, "from_last": True, "test_last": True}}}

        task = LAMAtask('multi:reg', greater_is_better=False, metric="mae", loss="mae")
        roles = {"target": TARGET}
        self.model = AutoTS(task, seq_params=seq_params, trend_params={"trend": True})
        train_pred, _ = self.model.fit_predict(train, roles)

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('LAMA does not support fit for multivariate time series forecasting')

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                              **kwargs) -> ForecastResults:
        """ Use obtained model to make predictions """
        # Update model weights
        historical_values['datetime'] = pd.to_datetime(historical_values['datetime'])
        train = historical_values.reset_index()[['datetime', 'value']]
        predicted_ts, _ = self.model.predict(train)
        return ForecastResults(predictions=predicted_ts)

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('LAMA does not support predict for multivariate time series forecasting')
