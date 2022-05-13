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
