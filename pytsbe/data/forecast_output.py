from typing import Optional, Any, Union, List

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ForecastResults:
    """ Dataclass for wrapping outputs from forecasting algorithm """

    # Predicted time series with 'datetime' and 'value' columns (for in-sample several dataframes returned)
    predictions: Union[np.ndarray, List[np.ndarray]] = None
    # Actual values for forecasted output
    true_values: Union[pd.DataFrame, List[pd.DataFrame]] = None
    # Timeouts after fit and predict in seconds
    timeouts: Optional[dict] = None
    # If there is an AutoML library, it is produced not only forecast but also the model
    obtained_model: Optional[Any] = None
    # Description of model, description of searching process, etc.
    additional_info: Optional[Any] = None

    @staticmethod
    def union(in_sample_results: list):
        """ Perform union of several forecasts """
        # Take unchanged attributes
        additional_info = in_sample_results[0].additional_info
        obtained_model = in_sample_results[0].obtained_model
        timeouts = in_sample_results[0].timeouts

        ts_dataframes = []
        all_predictions = []
        for i, forecast in enumerate(in_sample_results):
            # Take changed attributes
            ts_dataframe = forecast.true_values
            ts_dataframe['validation_block'] = [i] * len(ts_dataframe)

            ts_dataframes.append(ts_dataframe)
            all_predictions.append(np.ravel(forecast.predictions))

        ts_dataframes = pd.concat(ts_dataframes)
        all_predictions = np.hstack(all_predictions)

        return ForecastResults(predictions=all_predictions, true_values=ts_dataframes,
                               timeouts=timeouts, obtained_model=obtained_model,
                               additional_info=additional_info)
