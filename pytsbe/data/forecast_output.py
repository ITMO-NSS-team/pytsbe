from typing import Optional, Any, Union, List

import pandas as pd
from dataclasses import dataclass


@dataclass
class ForecastResults:
    """ Dataclass for wrapping outputs from forecasting algorithm """

    # Predicted time series with 'datetime' and 'value' columns (for in-sample several dataframes returned)
    predictions: Union[pd.DataFrame, List[pd.DataFrame]] = None
    # Actual values for forecasted output
    true_values: Union[pd.DataFrame, List[pd.DataFrame]] = None
    # If there is an AutoML library, it is produced not only forecast but also the model
    saved_models: Optional[Any] = None
    # Description of model
    model_info: Optional[Any] = None
    # Metadata about model search process
    model_search_process: Optional[Any] = None
