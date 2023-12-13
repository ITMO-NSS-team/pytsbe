from abc import abstractmethod

import pandas as pd

from pytsbe.data.forecast_output import ForecastResults


class Forecaster:
    """ The class implements a unified interface for generating forecasts """

    def __init__(self, **params):
        self.params = params
        self.model = None

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
        n_rows, n_cols = historical_values.shape

        if n_cols > 2:
            # Multivariate time series forecasting
            target_column, predictors_columns = find_target_and_exog_variables(historical_values)
            return self.fit_multivariate_ts(historical_values, forecast_horizon,
                                            target_column, predictors_columns, **kwargs)
        else:
            # Univariate time series forecasting
            return self.fit_univariate_ts(historical_values, forecast_horizon, **kwargs)

    @abstractmethod
    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        """ There is a needed to implement method to train model for forecasting univariate time series """
        raise NotImplementedError()


    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        """ There is a needed to implement method to train model for forecasting multivariate time series """
        raise NotImplementedError(f'{self.__class__} does not have method for multivariate ts')

    def predict(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs) -> ForecastResults:
        """ Generate predictions based on historical values for only one forecast horizon """
        n_rows, n_cols = historical_values.shape

        if n_cols > 2:
            # Multivariate time series forecasting
            target_column, predictors_columns = find_target_and_exog_variables(historical_values)
            return self.predict_multivariate_ts(historical_values, forecast_horizon,
                                                target_column, predictors_columns, **kwargs)
        else:
            # Univariate time series forecasting
            return self.predict_univariate_ts(historical_values, forecast_horizon, **kwargs)

    @abstractmethod
    def predict_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        raise NotImplementedError(f'{self.__class__} does not have method for multivariate ts')

    @abstractmethod
    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        """ Warning! In table target column will contain 'target' in column name """
        raise NotImplementedError()


def find_target_and_exog_variables(historical_values: pd.DataFrame):
    """ Find names of columns for multivariate time series """
    target_plus_exogenous_columns = list(historical_values.columns)
    target_column = list(filter(lambda x: 'target' in str(x), target_plus_exogenous_columns))
    target_column = str(target_column[0])
    target_plus_exogenous_columns.remove('datetime')
    return target_column, target_plus_exogenous_columns
