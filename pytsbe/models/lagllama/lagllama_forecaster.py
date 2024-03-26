import copy

import torch
import pandas as pd

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.model.forecast import Forecast
from pathlib import Path
from typing import List, Optional

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.models.forecast import Forecaster
from pytsbe.models.lagllama.lag_llama.lag_llama.gluon.estimator import LagLlamaEstimator
from pytsbe.paths import get_project_path


class LagLlamaForecaster(Forecaster):
    def __init__(self, **params):
        super().__init__(**params)
        self.target = 'value'
        self.estimator = None
        self.predictor = None

    def fit_univariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int, **kwargs):
        if self.estimator is None:
            self.estimator = self.__load_ckpt(forecast_horizon=forecast_horizon)
        lightning_module = self.estimator.create_lightning_module()
        transformation = self.estimator.create_transformation()
        self.predictor = self.estimator.create_predictor(transformation, lightning_module)

    def fit_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                            target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('Lag-Llama does not support fit for multivariate time series forecasting')

    def predict_univariate_ts(self, historical_values: pd.DataFrame,
                              forecast_horizon: int, **kwargs) -> ForecastResults:
        df = copy.deepcopy(historical_values)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        forecast: Optional[List[Forecast]]
        for col in df.columns:
            if df[col].dtype != 'object':
                if not pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].astype('float32')
        dataset = PandasDataset(df, target=self.target)

        forecast, _ = make_evaluation_predictions(
            dataset=dataset,
            predictor=self.predictor
        )
        predictions = [f.median for f in forecast]

        return ForecastResults(predictions=predictions[0].tolist())

    def predict_multivariate_ts(self, historical_values: pd.DataFrame, forecast_horizon: int,
                                target_column: str, predictors_columns: list, **kwargs):
        raise NotImplementedError('Lag-Llama does not support predict for multivariate time series forecasting')

    @staticmethod
    def __load_ckpt(forecast_horizon: int):
        ckpt_path = Path(get_project_path()).joinpath('pytsbe/models/lagllama/lag_llama/lag_llama.ckpt')
        estimator_args = torch.load(
            str(ckpt_path),
            map_location=torch.device('cpu')
        )['hyper_parameters']['model_kwargs']
        return LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=forecast_horizon,
            context_length=32,
            input_size=estimator_args['input_size'],
            n_layer=estimator_args['n_layer'],
            n_embd_per_head=estimator_args['n_embd_per_head'],
            n_head=estimator_args['n_head'],
            scaling=estimator_args['scaling'],
            time_feat=estimator_args['time_feat'],
            batch_size=1
        )
