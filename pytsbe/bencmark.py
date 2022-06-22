import os
import yaml

from pytsbe.main import TimeSeriesLauncher


# Configuration for experiment
from pytsbe.report.report import MetricsReport

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
            config_path = os.path.join(os.path.curdir, 'configuration.yaml')
        self.config_path = os.path.abspath(config_path)

        # Create object for experiments
        self.working_dir = os.path.abspath(working_dir)
        self.experimenter = TimeSeriesLauncher(working_dir=working_dir,
                                               datasets=DATASETS,
                                               launches=LAUNCHES)

    def run(self, file_name: str = None):
        """ Start experiment with desired configuration

        :param file_name: name of csv file to save aggregated final metrics
        """
        libraries_params = self.get_libraries_info()
        libraries_to_compare = list(libraries_params.keys())
        libraries_to_compare.sort()

        # Start experiments
        self.experimenter.perform_experiment(libraries_to_compare=libraries_to_compare,
                                             libraries_params=libraries_params,
                                             horizons=HORIZONS,
                                             validation_blocks=VALIDATION_BLOCKS,
                                             clip_border=CLIP_BORDER)

        # Collect reports with execution times and SMAPE metric
        metrics_processor = MetricsReport(working_dir=self.working_dir)
        timeouts_table = metrics_processor.time_execution_table(aggregation=['Library'])
        metrics_table = metrics_processor.metric_table(metrics=['SMAPE'], aggregation=['Library'])

        # Aggregated metrics
        final_metrics = metrics_table.merge(timeouts_table, on='Library')
        print('Final metrics:')
        print(final_metrics)

        if file_name is None:
            file_name = 'univariate_benchmark_metrics.csv'
        final_metrics.to_csv(file_name, index=False)
        return final_metrics

    def get_libraries_info(self) -> dict:
        """ Get libraries parameters from configuration files """
        with open(self.config_path) as file:
            configuration = yaml.load(file, Loader=yaml.FullLoader)

        return configuration
