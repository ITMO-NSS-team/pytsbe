import pandas as pd
from typing import Optional, List, Union

from pytsbe.data.data import TimeSeriesDatasets
from pytsbe.models.fedot_forecaster import FedotForecaster
from pytsbe.data.forecast_output import ForecastResults


class Validator:
    """ Class for validation on only one selected dataset for the required forecast horizons

    Important: responsible for time series (from datasets) and horizons cycles
    """
    forecaster_by_name = {'FEDOT': FedotForecaster,
                          'AutoTS': None,
                          'TPOT': None,
                          'H2O': None}

    def __init__(self, library_name: str, library_parameters: dict, library_serializer):
        if library_name not in self.forecaster_by_name:
            raise NotImplementedError(f'Library {library_name} is not supported yet')

        self.library_name = library_name
        self.library_parameters = library_parameters
        self.library_serializer = library_serializer

    def run_all_experiment(self, dataset: TimeSeriesDatasets,
                           horizons: List[int],
                           validation_blocks: Optional[int] = None):
        """ Launch experiment with desired dataset and time series forecasting algorithms """
        for i, time_series in enumerate(dataset.time_series):
            ts_label = dataset.labels[i]

            forecaster = self.forecaster_by_name[self.library_name](**self.library_parameters)
            for horizon in horizons:
                # Prepare model for current forecast horizon
                results = self._perform_single_experiment(forecaster, time_series, horizon, validation_blocks)

                # Save all the necessary results
                self.library_serializer.save_information(ts_label, results)

    def _perform_single_experiment(self, forecaster, time_series: pd.DataFrame,
                                   horizon: int, validation_blocks: Union[int, None]) -> ForecastResults:
        """ Launch time series forecasting algorithm on single time series for particular """
        if validation_blocks is None or validation_blocks == 1:
            # Simple experiment - predict on only one fold
            train_values, historical_values_for_test, test_values = simple_train_test_split(time_series, horizon)

            forecaster.fit(train_values)
            forecast_output = forecaster.predict(historical_values=historical_values_for_test,
                                                 forecast_horizon=horizon)
            forecast_output.true_values = forecast_output
        else:
            # In-sample validation required
            train_values, historical_values_for_test, test_values = in_sample_train_test_split(time_series,
                                                                                               horizon,
                                                                                               validation_blocks)


def simple_train_test_split(time_series: pd.DataFrame, horizon: int):
    """ Perform simple train test split for time series to make forecast on only one fold """
    train_len = len(time_series) - horizon

    train_values = time_series.head(train_len)
    historical_values_for_test = train_values.copy()
    test_values = time_series.tail(horizon)
    return train_values, historical_values_for_test, test_values


def in_sample_train_test_split(time_series: pd.DataFrame, horizon: int, validation_blocks: int):
    raise NotImplementedError()
