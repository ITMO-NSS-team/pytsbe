from abc import abstractmethod

import pandas as pd

from pytsbe.data.forecast_output import ForecastResults


class Forecaster:
    """ The class implements a unified interface for generating forecasts """

    def __init__(self, **params):
        self.params = params
        self.model = None

    @abstractmethod
    def fit(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ Fit model (or library) with desired parameters

        :param historical_values: dataframe with datetime column and target series.
        For example:
        |  datetime  | value |
        | 01-01-2022 |  254  |
        | 02-01-2022 |  223  |
        :param forecast_horizon: forecast length
        :param kwargs: additional parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs) -> ForecastResults:
        """ Generate predictions based on historical values for only one forecast horizon """
        raise NotImplementedError()
