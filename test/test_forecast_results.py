import numpy as np
import pandas as pd

from pytsbe.data.forecast_output import ForecastResults


def generate_list_with_in_sample_results():
    """ During in-sample validation algorithm produce forecast at each step """
    first_df = pd.DataFrame({'datetime': ['2016-10-01', '2016-10-02'], 'value': [100, 110]})
    first_df['datetime'] = pd.to_datetime(first_df['datetime'])
    first_sample_output = ForecastResults(predictions=np.array([10, 11]), true_values=first_df,
                                          timeouts={'fit_seconds': 0.5, 'predict_seconds': 0.1})

    second_df = pd.DataFrame({'datetime': ['2016-10-03', '2016-10-04'], 'value': [120, 130]})
    second_df['datetime'] = pd.to_datetime(second_df['datetime'])
    second_sample_output = ForecastResults(predictions=np.array([12, 13]), true_values=second_df,
                                           timeouts={'fit_seconds': 0.5, 'predict_seconds': 0.1})

    return [first_sample_output, second_sample_output]


def test_forecast_union_after_in_sample():
    """ Check if algorithm for time series union work correctly """
    # Generate results from algorithm
    results = generate_list_with_in_sample_results()
    forecast_output = ForecastResults.union(results)

    assert np.array_equal(forecast_output.predictions, np.array([10, 11, 12, 13]))
    assert np.isclose(forecast_output.timeouts['fit_seconds'], 0.5)
    assert len(forecast_output.true_values) == 4
