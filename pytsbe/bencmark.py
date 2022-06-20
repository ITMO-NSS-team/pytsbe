import os
import yaml

from pytsbe.main import TimeSeriesLauncher


DATASETS = ['FRED', 'SMART', 'TEP']
LAUNCHES = 5
CLIP_BORDER = 2000
VALIDATION_BLOCKS = 3
HORIZONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class BenchmarkUnivariate:
    """
    Class for benchmarking different time series forecasting algorithms on
    univariate time series
    """

    def __init__(self, working_dir: str, config_path: str = None):
        if config_path is None:
            # Search for configuration path
            config_path = os.path.join(os.path.curdir, 'default_configuration.yaml')
        self.config_path = os.path.abspath(config_path)

        # Create object for experiments
        self.experimenter = TimeSeriesLauncher(working_dir=working_dir,
                                               datasets=DATASETS,
                                               launches=LAUNCHES)

    def run(self):
        """ Start experiment with desired configuration """
        libraries_params = self.get_libraries_info()
        libraries_to_compare = list(libraries_params.keys())
        libraries_to_compare.sort()

        self.experimenter.perform_experiment(libraries_to_compare=libraries_to_compare,
                                             libraries_params=libraries_params,
                                             horizons=HORIZONS,
                                             validation_blocks=VALIDATION_BLOCKS,
                                             clip_border=CLIP_BORDER)

        # TODO: launch plots composing

    def get_libraries_info(self) -> dict:
        """ Get libraries parameters from configuration files """
        with open(self.config_path) as file:
            configuration = yaml.load(file, Loader=yaml.FullLoader)

        return configuration
