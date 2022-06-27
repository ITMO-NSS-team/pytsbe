import json
import os
import yaml
from typing import List

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
        if self.is_new_libraries_added():
            self.add_libraries_to_configuration()

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

    def is_new_libraries_added(self):
        """ Check if new libraries were added into configuration file """
        if os.path.isfile(self.experimenter.path_to_config_json) is False:
            # It is first launch
            return False

        with open(self.experimenter.path_to_config_json) as file:
            config_info = json.load(file)

        old_libraries = set(config_info['Libraries to compare'])
        new_libraries = set(self.configuration['libraries'].keys())
        difference = new_libraries - old_libraries

        return len(difference) > 0

    def add_libraries_to_configuration(self):
        """ Add new libraries to configuration json file """
        with open(self.experimenter.path_to_config_json) as file:
            config_info = json.load(file)

        libraries = config_info['Libraries to compare']
        current_libraries = list(self.configuration['libraries'].keys())
        new_libraries = list(set(current_libraries) - set(libraries))
        print(f'The following libraries were added: {new_libraries}')

        # Update configuration file
        config_info['Libraries to compare'] = current_libraries
        libraries_params = config_info['Libraries parameters']
        for new_library in new_libraries:
            new_lib_parameters = self.configuration['libraries'][new_library]
            libraries_params.update({new_library: new_lib_parameters})

        with open(self.experimenter.path_to_config_json, 'w') as file:
            json.dump(config_info, file)


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
