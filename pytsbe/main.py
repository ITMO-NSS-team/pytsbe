import os
import json
from itertools import product
from typing import List, Optional

from pytsbe.data.data import TimeSeriesDatasets
from pytsbe.store.save import Serialization
from pytsbe.validation.validation import Validator


class TimeSeriesLauncher:
    """ Class for performing experiments for time series forecasting task.
    Launch different forecasting libraries or algorithms through one interface.

    Important: responsible for datasets, launch and libraries cycles
    :param working_dir: directory where to store the results of experiments
    :param datasets: list with names of datasets to perform validation
    :param launches: number of launches for each dataset
    """

    def __init__(self, working_dir: str, datasets: List[str], launches: int = 1):
        self.serializer = Serialization(working_dir)
        self.datasets = datasets
        self.launches = launches

        self.path_to_config_json = os.path.join(working_dir, 'configuration.json')

    def perform_experiment(self,
                           libraries_to_compare: List[str],
                           horizons: List[int],
                           libraries_params: dict = None,
                           validation_blocks: Optional[int] = None,
                           clip_border: int = None):
        """ Perform time series experiments with desired libraries

        :param libraries_to_compare: list with libraries for comparison
        :param horizons: forecast horizons to process
        :param libraries_params: parameters for libraries
        :param validation_blocks: validation blocks for in-sample forecasting
        :param clip_border: is there a need to clip time series (if None - there is no cropping)
        """
        if os.path.exists(self.path_to_config_json):
            # Experiments hase been previously configured - check the relevance
            self.check_experiment_configuration(libraries_to_compare, horizons,
                                                libraries_params, validation_blocks, clip_border)
        else:
            # Configure new experiment
            self.store_experiment_configuration(libraries_to_compare, horizons,
                                                libraries_params, validation_blocks, clip_border)
        self.serializer.create_folders_for_results(self.datasets, self.launches, libraries_to_compare)

        for dataset_name in self.datasets:
            # Prepare data in pytsbe dataset form
            dataset = TimeSeriesDatasets.configure_dataset_from_path(dataset_name=dataset_name,
                                                                     clip_border=clip_border)

            experiments = product(range(self.launches), libraries_to_compare)
            for launch_number, current_library_name in experiments:
                print(f'Dataset {dataset_name} launch number {launch_number} for library {current_library_name}')

                # Get helper for serialization procedures for appropriate library
                current_library_serializer = self.serializer.get(current_library_name)
                current_library_serializer.set_configuration_params(dataset_name, launch_number, current_library_name)

                # Configure validation module and perform experiments
                current_library_parameters = libraries_params.get(current_library_name)
                validator = Validator(dataset_name, launch_number, current_library_name,
                                      current_library_parameters, current_library_serializer)
                validator.perform_experiments_on_dataset_and_horizons(dataset=dataset,
                                                                      horizons=horizons,
                                                                      validation_blocks=validation_blocks)

    def store_experiment_configuration(self, libraries_to_compare, horizons,
                                       libraries_params, validation_blocks,
                                       clip_border):
        """ Save useful information about experiment into json file """
        info_to_save = params_into_dict(self.datasets, self.launches, libraries_to_compare,
                                        horizons, libraries_params, validation_blocks, clip_border)

        with open(self.path_to_config_json, 'w') as file:
            json.dump(info_to_save, file)

    def check_experiment_configuration(self, libraries_to_compare, horizons,
                                       libraries_params, validation_blocks,
                                       clip_border):
        """ Compare existing configuration file and current configuration """
        message = 'New and old experiment configurations differ! Roll back' \
                  ' configuration or create new directory for experiments.'

        config_info = params_into_dict(self.datasets, self.launches, libraries_to_compare,
                                        horizons, libraries_params, validation_blocks, clip_border)
        with open(self.path_to_config_json) as file:
            old_config_info = json.load(file)

        if tuple(config_info.keys()) != tuple(old_config_info.keys()):
            raise ValueError(message)

        for parameter_name, value in config_info.items():
            old_value = old_config_info[parameter_name]
            # Compare each parameter
            if isinstance(value, list):
                # Order of parameters not so important
                if set(value) != set(old_value):
                    raise ValueError(message)
            else:
                # It's dict pr value (int or float)
                if value != old_value:
                    raise ValueError(message)


def params_into_dict(datasets, launches, libraries_to_compare, horizons,
                     libraries_params, validation_blocks, clip_border):
    config_info = {'Datasets': datasets,
                   'Launches': launches,
                   'Libraries to compare': libraries_to_compare,
                   'Libraries parameters': libraries_params,
                   'Horizons': horizons,
                   'Validation blocks': validation_blocks,
                   'Clip border': clip_border}
    return config_info
