import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNBEATS, AutoMLP
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import NBEATS
from neuralforecast.tsdataset import TimeSeriesDataset

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster

import logging

from pytsbe.models.nn_forecasters.auto_nn_forecaster import NNForecaster

logging.raiseExceptions = False


class MLPForecaster(NNForecaster):
    """
    Class for time series forecasting with AutoMLP
    Source code: https://nixtlaverse.nixtla.io/neuralforecast/models.html#automlp
    """

    def _init_model(self, forecast_horizon):
        gpus = int(self.accelerator == 'gpu')
        model = AutoMLP(h=forecast_horizon, gpus=gpus, backend='optuna')
        return model
