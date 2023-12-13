from typing import Callable

import pytest

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.autogluon_forecater import AutoGluonForecaster
from pytsbe.models.average_forecaster import NaiveAverageForecaster
from pytsbe.models.fedot_forecaster import FedotForecaster
from pytsbe.models.lama_forecaster import LAMAForecaster
from pytsbe.models.naive_repeat_forecaster import NaiveRepeatLastValueForecaster
from pytsbe.models.nn_forecasters.auto_timesnet_forecater import NBEATSForecaster
from test.test_data import get_dummy_dataset


def get_univariate_time_series():
    """ Return first time series from dummy dataset (univariate case) """
    dataset = get_dummy_dataset('dummy')
    return dataset.get_time_series_by_label('series_1')


def get_multivariate_time_series():
    """ Return first time series from dummy dataset (multivariate case) """
    dataset = get_dummy_dataset('multivariate_dummy')
    for i, ts in enumerate(dataset.time_series):
        if i == 0:
            # Corrupt time series a little bit
            ts['target_series_1'].iloc[10] = 50
            return ts


@pytest.mark.parametrize('forecaster, forecaster_params',
                         [(LAMAForecaster, {}),
                          (NaiveAverageForecaster, {}),
                          (NaiveRepeatLastValueForecaster, {}),
                          (FedotForecaster, {'predefined_model': 'ar'}),
                          (AutoGluonForecaster, {}),
                          (NBEATSForecaster, {})])
def test_univariate_models(forecaster: Callable, forecaster_params: dict):
    """ Automatically testing univariate forecasting models """
    forecast_horizon = 10
    time_series = get_univariate_time_series()

    forecaster = forecaster(**forecaster_params)

    forecaster.fit(historical_values=time_series, forecast_horizon=forecast_horizon)
    predict = forecaster.predict(historical_values=time_series, forecast_horizon=forecast_horizon)

    assert isinstance(predict, ForecastResults)
    assert len(predict.predictions) == forecast_horizon


@pytest.mark.parametrize('forecaster, forecaster_params',
                         [(NaiveAverageForecaster, {}),
                          (NaiveRepeatLastValueForecaster, {}),
                          (FedotForecaster, {'predefined_model': 'auto'})])
def test_multivariate_models(forecaster: Callable, forecaster_params: dict):
    """ Testing multivariate forecasting models """
    forecast_horizon = 10
    time_series = get_multivariate_time_series()

    forecaster = forecaster(**forecaster_params)

    forecaster.fit(historical_values=time_series, forecast_horizon=forecast_horizon)
    predict = forecaster.predict(historical_values=time_series, forecast_horizon=forecast_horizon)

    assert isinstance(predict, ForecastResults)
    assert len(predict.predictions) == forecast_horizon
