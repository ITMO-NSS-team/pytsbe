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
                current_library_parameters = libraries_params[current_library_name]
                validator = Validator(current_library_name, current_library_parameters, current_library_serializer)
                validator.perform_experiments_on_dataset_and_horizons(dataset=dataset,
                                                                      horizons=horizons,
                                                                      validation_blocks=validation_blocks)
