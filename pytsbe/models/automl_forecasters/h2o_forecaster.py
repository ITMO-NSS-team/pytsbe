from copy import copy

from pytsbe.models.automl_forecasters.automl import AutoMLForecaster, TIMEOUT_RATIO_FOR_TUNING

try:
    import tpot
    import h2o
    from fedot.api.main import Fedot
    from fedot.core.data.data import InputData
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.repository.operation_types_repository import OperationTypesRepository
except ImportError:
    print('Does not found FEDOT for H2O, tpot libraries launch and / or H2O, tpot libraries. Continue...')


class H2OForecaster(AutoMLForecaster):
    """
    Class for time series forecasting with H2O AutoML framework. Launch as a
    part of FEDOT framework
    Source code: https://github.com/h2oai/h2o-3
    """

    def __init__(self, **params):
        super().__init__(**params)
        default_params = {'timeout': 2, 'max_models': 10}
        if params is not None:
            self.params = {**default_params, **params}
        else:
            self.params = default_params
        self.obtained_pipeline = None

        # Determine timeout for tuning lagged transformation hyperparameters
        self.timeout_for_tuning = round(TIMEOUT_RATIO_FOR_TUNING * self.params['timeout'])
        if self.timeout_for_tuning == 0:
            # Give at least 30 seconds for tuning
            self.timeout_for_tuning = 0.5
        self.remained_timeout = round(self.params['timeout'] - self.timeout_for_tuning)
        if self.remained_timeout <= 0:
            self.remained_timeout = 1

    def _configure_automl_pipeline(self, window_size):
        """ Create pipeline for time series forecasting with AutoML model as final model """
        lagged_node = PrimaryNode('lagged')
        lagged_node.custom_params = {'window_size': window_size}
        h2o_node = SecondaryNode('h2o_regr', nodes_from=[lagged_node])

        automl_params = copy(self.params)
        automl_params.update({'timeout': self.remained_timeout})

        h2o_node.custom_params = automl_params
        pipeline = Pipeline(h2o_node)

        return pipeline
