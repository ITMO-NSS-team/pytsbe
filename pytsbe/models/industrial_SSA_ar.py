import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.core.operation.transformation.WindowSelection import WindowSizeSelection

try:
    from fedot.api.main import Fedot
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import TsForecastingParams, TaskTypesEnum, Task
except ImportError:
    print('Does not found FEDOT library. Continue...')

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging

logging.raiseExceptions = False


class Industrial_SSA_ar(Forecaster):
    """
    Class for time series forecasting with FEDOT framework
    Source code:
    """

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, target_column: str,
                            predictors_columns: list, **kwargs):
        pass

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, target_column: str,
                                predictors_columns: list, **kwargs):
        pass

    def __init__(self, **params):
        super().__init__(**params)
        default_params = {'n_components': 5}
        if params is not None:
            self.params = {**default_params, **params}
        else:
            self.params = default_params

        self.obtained_pipeline = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Train FEDOT framework (launch AutoML algorithm) """
        train_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=False)
        n, _ = WindowSizeSelection(time_series=train_data.features,
                                   wss_algorithm='dominant_fourier_frequency').get_window_size()

        self.obtained_pipeline = PipelineBuilder().add_node('data_driven_basis_for_forecasting',
                                                            params={'n_components': 6, 'window_size': n,
                                                                    'estimator': PipelineBuilder().add_node(
                                                                        'ar').build()
                                                                    },
                                                            ).build()

        self.obtained_pipeline.fit(train_data)

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Use obtained pipeline to make predictions """

        historical_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=True)
        forecast = self.obtained_pipeline.predict(historical_data).predict

        result = ForecastResults(predictions=forecast, obtained_model=self.obtained_pipeline)
        return result


def prepare_input_ts_data(historical_values: pd.DataFrame, forecast_horizon: int,
                          is_for_forecast: bool = False):
    """ Return converted into InputData datasets for train and for prediction """
    time_series_label = 'value'
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_horizon))

    if is_for_forecast:
        start_forecast = len(historical_values)
        end_forecast = start_forecast + forecast_horizon

        input_data = InputData(idx=np.arange(start_forecast, end_forecast),
                               features=np.array(historical_values[time_series_label]),
                               target=np.array(historical_values[time_series_label]),
                               task=task, data_type=DataTypesEnum.ts)
    else:
        input_data = InputData(idx=np.arange(0, len(historical_values)),
                               features=np.array(historical_values[time_series_label]),
                               target=np.array(historical_values[time_series_label]),
                               task=task, data_type=DataTypesEnum.ts)

    return input_data
