from abc import abstractmethod
from datetime import timedelta

import numpy as np
import pandas as pd
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.fedot_forecaster import get_simple_pipeline, prepare_input_ts_data
from pytsbe.models.forecast import Forecaster

try:
    import tpot
    import h2o
    from fedot.api.main import Fedot
    from fedot.core.data.data import InputData
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.repository.operation_types_repository import OperationTypesRepository
except ImportError:
    print('Does not found FEDOT for H2O, tpot libraries launch and / or H2O, tpot libraries. Continue...')

TIMEOUT_RATIO_FOR_TUNING = 0.5


class AutoMLForecaster(Forecaster):
    """
    Class for time series forecasting with AutoML frameworks. Launch as a
    part of FEDOT framework. First a simple pipeline with lagged transformation
    and decision tree regression model is generated. Then the hyperparameters
    will be configured (select the appropriate window size for lagged). Then
    the final model will be replaced by AutoML (TPOT or H2o for example)
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.obtained_pipeline = None
        self.timeout_for_tuning = None
        self.remained_timeout = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        try:
            with OperationTypesRepository.init_automl_repository() as repo:
                train_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=False)
                self.obtained_pipeline = self._substitute_automl_training(train_data)
        except AttributeError:
            # Repository has already been initialized
            train_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=False)
            self.obtained_pipeline = self._substitute_automl_training(train_data)

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('AutoML tools (TPOT and H2O) does not support fit '
                                  'for multivariate time series forecasting')

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                              **kwargs) -> ForecastResults:
        """ Use fitted model to prepare forecast """
        historical_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=True)
        forecast = self.obtained_pipeline.predict(historical_data)

        lagged_params = self.obtained_pipeline.nodes[-1].parameters
        window_size = lagged_params['window_size']

        return ForecastResults(predictions=np.ravel(np.array(forecast.predict)),
                               additional_info={'lagged_window_size': window_size})

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('AutoML tools (TPOT and H2O) does not support predict '
                                  'for multivariate time series forecasting')

    def _substitute_automl_training(self, input_data: 'InputData'):
        """
        Using simple pipeline, tune lagged operation and then replace final model (decision tree)
        with AutoML as operation in the node
        """
        simple_pipeline = get_simple_pipeline()

        tuner = TunerBuilder(input_data.task) \
            .with_tuner(SimultaneousTuner) \
            .with_metric(RegressionMetricsEnum.MAE) \
            .with_iterations(500) \
            .with_timeout(timedelta(seconds=self.timeout_for_tuning)) \
            .build(input_data)

        pipeline = tuner.tune(simple_pipeline)

        lagged_params = pipeline.nodes[-1].parameters
        window_size = lagged_params['window_size']
        # Get new pipeline with AutoML as operation
        automl_pipeline = self._configure_automl_pipeline(window_size)
        automl_pipeline.fit(input_data)

        return automl_pipeline

    @abstractmethod
    def _configure_automl_pipeline(self, window_size):
        """ Create pipeline for time series forecasting with AutoML model as final model """
        raise NotImplementedError()
