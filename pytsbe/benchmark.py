import os
import yaml

from pytsbe.main import TimeSeriesLauncher
from pytsbe.report.report import MetricsReport


class Benchmark:
    """ Base class for benchmarking """

    def __init__(self, working_dir: str, config_path: str = None):
        if config_path is None:
            # Search for configuration path
            config_path = os.path.join(os.path.curdir, 'configuration.yaml')
        self.config_path = os.path.abspath(config_path)

        # Read configuration file
        with open(self.config_path) as file:
            self.configuration = yaml.load(file, Loader=yaml.FullLoader)

        # Create object for experiments
        self.working_dir = os.path.abspath(working_dir)
        self.experimenter = TimeSeriesLauncher(working_dir=working_dir,
                                               datasets=self.configuration['datasets'],
                                               launches=self.configuration['launches'])
        self.default_file_name = 'benchmark_metrics.csv'

    def run(self, file_name: str = None):
        """ Start experiment with desired configuration

        :param file_name: name of csv file to save aggregated final metrics
        """
        libraries_params = self.configuration['libraries']
        libraries_to_compare = list(libraries_params.keys())
        libraries_to_compare.sort()

        # Start experiments
        self.experimenter.perform_experiment(libraries_to_compare=libraries_to_compare,
                                             libraries_params=libraries_params,
                                             horizons=self.configuration['horizons'],
                                             validation_blocks=self.configuration['validation_blocks'],
                                             clip_border=self.configuration['clip_border'])

        # Collect reports with execution times and SMAPE metric
        metrics_processor = MetricsReport(working_dir=self.working_dir)
        timeouts_table = metrics_processor.time_execution_table(aggregation=['Library'])
        metrics_table = metrics_processor.metric_table(metrics=['SMAPE'], aggregation=['Library'])

        # Aggregated metrics
        final_metrics = metrics_table.merge(timeouts_table, on='Library')
        print('Final metrics:')
        print(final_metrics)

        if file_name is None:
            file_name = self.default_file_name
        final_metrics.to_csv(file_name, index=False)
        return final_metrics


class BenchmarkUnivariate(Benchmark):
    """
    Class for benchmarking different time series forecasting algorithms on
    univariate time series
    """

    def __init__(self, working_dir: str, config_path: str = None):
        super().__init__(working_dir, config_path)
        self.default_file_name = 'univariate_benchmark_metrics.csv'


class BenchmarkMultivariate(Benchmark):
    """
    Class for benchmarking different time series forecasting algorithms on
    multivariate time series
    """

    def __init__(self, working_dir: str, config_path: str = None):
        super().__init__(working_dir, config_path)
        self.default_file_name = 'multivariate_benchmark_metrics.csv'