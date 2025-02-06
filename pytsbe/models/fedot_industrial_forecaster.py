import pandas as pd
import numpy as np
import shutil

try:
    from fedot_ind.api.main import FedotIndustrial
    from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG
except ImportError:
    print('Does not found Fedot.Industrial library. Continue...')


from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging
logging.raiseExceptions = False


class FedotIndustrialForecaster(Forecaster):
    """
    Class for time series forecasting with FEDOT.Industrial framework
    Source code: https://github.com/aimclub/Fedot.Industrial
    """

    def __init__(self, **params):
        super().__init__(**params)
        default_api_config = init_default_config()
        self.init_params = {**default_api_config, **params}
        self.obtained_model = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Train FEDOT.Industrial framework (launch AutoML algorithm) """
        input_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=False)

        task_params = {'forecast_length': forecast_horizon}
        self.init_params['industrial_config']['task_params'] = task_params
        self.init_params['automl_config']['task_params'] = task_params

        model = FedotIndustrial(**self.init_params)
        model.fit(input_data)
        self.obtained_model = model
        self.obtained_model.shutdown()
        # # TODO: remove when composition history managing becomes a responsibility of Fedot.Industrial
        # shutil.rmtree(model.config_dict.get('history_dir'))

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        """ Create pipeline for multivariate time series forecasting """
        raise NotImplementedError()

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Use obtained pipeline to make predictions """
        input_data = prepare_input_ts_data(historical_values, forecast_horizon, is_for_forecast=True)
        labels = self.obtained_model.predict(input_data)
        return ForecastResults(predictions=labels)

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError()


def prepare_input_ts_data(historical_values: pd.DataFrame, forecast_horizon: int, is_for_forecast: bool):
    """ Return converted into InputData datasets for train and for prediction """
    time_series_label = 'value'
    series = np.array(historical_values[time_series_label]).flatten()
    if is_for_forecast:
        return series, series
    return series, series[-forecast_horizon:]


def init_default_config():
    COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
    COMPUTE_CONFIG['distributed'] = dict(processes=False,
                                         n_workers=2,
                                         threads_per_worker=2,
                                         memory_limit=0.3)
    AUTOML_CONFIG = {'task': 'ts_forecasting',
                     'task_params': {'forecast_length': 1},
                     'use_automl': True,
                     'optimisation_strategy': {'optimisation_strategy':
                                                   {'mutation_agent': 'random',
                                                    'mutation_strategy': 'growth_mutation_strategy'},
                                               'optimisation_agent': 'Industrial'}}
    AUTOML_LEARNING_STRATEGY = dict(timeout=5,
                                    n_jobs=4,
                                    pop_size=10,
                                    with_tuning=False,
                                    logging_level=20)
    LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                       'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                       'optimisation_loss': {'quality_loss': 'rmse'}}
    INDUSTRIAL_CONFIG = {'problem': 'ts_forecasting',
                         'task_params': {'forecast_length': 1}}

    return {'industrial_config': INDUSTRIAL_CONFIG,
            'automl_config': AUTOML_CONFIG,
            'learning_config': LEARNING_CONFIG,
            'compute_config': COMPUTE_CONFIG}
