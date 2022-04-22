import os
from typing import List, Optional

from pytsbe.data import TimeSeriesDatasets
from pytsbe.launch.run_autots import AutoTSTsRun
from pytsbe.launch.run_fedot import FedotTsRun
from pytsbe.launch.run_automl import TPOTTsRun, H2OTsRun


class TimeSeriesLauncher:
    """ Class for performing experiments for time series forecasting task.
    Launch different forecasting libraries or algorithms through one interface
    """
    _ts_libraries = {'FEDOT': FedotTsRun,
                     'AutoTS': AutoTSTsRun,
                     'TPOT': TPOTTsRun,
                     'H2O': H2OTsRun,
                     'pmdarima': None,
                     'prophet': None,
                     'last': None,
                     'average': None,
                     }

    def __init__(self, working_dir, datasets: List[str], launches: int = 1):
        self.working_dir = working_dir
        self.datasets = datasets
        self.launches = launches

    def ts_experiment(self, libraries_to_compare: List[str], horizons: List[int], libraries_params: dict = None,
                      validation_blocks: Optional[int] = None):
        """ Perform time series experiments

        :param libraries_to_compare: list with libraries for comparison
        :param horizons: forecast horizons to process
        :param libraries_params: parameters for libraries
        :param validation_blocks: validation blocks for in-sample forecasting
        """

        for dataset_name, dataset_info in self.datasets_info.items():
            print(f'Dataset {dataset_name}')
            dataset_dir = os.path.join(self.working_dir, dataset_name)

            dataset_path = dataset_info['path']
            dataset_format = dataset_info['dataset_format']
            clip_to = dataset_info.get('clip_to')

            if dataset_format == 'wide':
                val_set = TimeSeriesDatasets.setup_from_wide_format(path=dataset_path, clip_to=clip_to)
            else:
                val_set = TimeSeriesDatasets.setup_from_long_format(path=dataset_path, clip_to=clip_to)

            for library in libraries_to_compare:
                print(f'Library {library}')
                # For every library start validation
                runner_class = self._ts_libraries[library]
                runner_params = libraries_params.get(library)

                library_dir = os.path.join(dataset_dir, library)
                runner = runner_class(val_set=val_set, working_dir=library_dir,
                                      params=runner_params, launches=self.launches)
                runner.perform_validation(horizons=horizons,
                                          validation_blocks=validation_blocks)
