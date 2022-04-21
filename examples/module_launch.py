import os

from pytsbe.paths import get_data_path
from pytsbe.main import TimeSeriesLauncher


if __name__ == '__main__':
    # For TEP and FRED datasets perform launches
    # working_dir - directory for saving results
    # datasets_info - dictionary with information about datasets
    #       * path - path to the csv file
    #       * dataset_format - wide or long format for time series
    #       * clip_to - how many elements should contain time series
    # launches - number of launches for averaging
    exp = TimeSeriesLauncher(working_dir='.',
                             datasets_info={'tep': {'path': os.path.join(get_data_path(), 'tep.csv'),
                                                    'dataset_format': 'wide',
                                                    'clip_to': 3000},
                                            'fred_long': {'path': os.path.join(get_data_path(), 'fred.csv'),
                                                          'dataset_format': 'long',
                                                          'clip_to': 3000}
                                            },
                             launches=5)
    exp.ts_experiment(libraries_to_compare=['FEDOT', 'AutoTS', 'H2O', 'TPOT'],
                      horizons=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      libraries_params={'H2O': {'timeout': 15, 'max_models': 15},
                                        'TPOT': {'timeout': 15, 'generations': 150, 'population_size': 20},
                                        'FEDOT': {'preset': 'ts', 'timeout': 15},
                                        'AutoTS': {'frequency': 'infer', 'prediction_interval': 0.9,
                                                   'ensemble': 'all', 'model_list': 'default',
                                                   'max_generations': 20, 'num_validations': 3}},
                      validation_blocks=2)
