import pandas as pd
import numpy as np
import shutil


try:
    from fedot_ind.api.main import FedotIndustrial
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
        default_params = {
            'timeout': 6,
            'n_jobs': 1,
            'metric': 'smape',
            'pop_size': 10,
            'with_tuning': False,
            'industrial_strategy': 'forecasting_assumptions'
        }
        self.init_params = {**default_params, **params}
        self.obtained_model = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Train FEDOT.Industrial framework (launch AutoML algorithm) """
        input_data = prepare_input_ts_data(historical_values, forecast_horizon)

        model = FedotIndustrial(problem='ts_forecasting',
                                task_params={'forecast_length': forecast_horizon},
                                **self.init_params)
        model.fit(input_data)
        self.obtained_model = model

        # TODO: remove when composition history managing becomes a responsibility of Fedot.Industrial
        shutil.rmtree(model.config_dict.get('history_dir'))

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        """ Create pipeline for multivariate time series forecasting """
        raise NotImplementedError()

    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Use obtained pipeline to make predictions """
        input_data = prepare_input_ts_data(historical_values, forecast_horizon)
        auto_labels = self.obtained_model.predict(input_data)

        min_metric = float('inf')
        metric = self.init_params.get('metric', 'smape')
        for forecast_model, predict in auto_labels.items():
            self.obtained_model.predicted_labels = predict
            current_metric = self.obtained_model.get_metrics(target=input_data[1],
                                                             metric_names=tuple([metric]))[metric][0]

            if float(current_metric) < min_metric:
                min_metric = current_metric
                forecast = predict

        return ForecastResults(predictions=np.array(forecast))

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError()


def prepare_input_ts_data(historical_values: pd.DataFrame, forecast_horizon: int):
    """ Return converted into InputData datasets for train and for prediction """
    time_series_label = 'value'
    series = np.array(historical_values[time_series_label])
    return series.flatten(), series[-forecast_horizon:].flatten()
