from pytsbe.main import TimeSeriesLauncher


def launch_example():
    """
    For FRED dataset perform 2 launches
    working_dir - directory for saving results
    datasets_info - dictionary with information about datasets
    launches - number of launches for averaging
    """
    experimenter = TimeSeriesLauncher(working_dir='.',
                                      datasets=['FRED', 'TEP', 'SMART'],
                                      launches=5)

    experimenter.perform_experiment(libraries_to_compare=['FEDOT', 'AutoTS', 'H2O', 'TPOT'],
                                    horizons=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                    libraries_params={'H2O': {'timeout': 2, 'max_models': 10},
                                                 'TPOT': {'timeout': 2, 'generations': 50, 'population_size': 16},
                                                 'FEDOT': {'preset': 'ts', 'timeout': 2},
                                                 'AutoTS': {'frequency': 'infer', 'prediction_interval': 0.9,
                                                            'ensemble': 'all', 'model_list': 'default',
                                                            'max_generations': 10, 'num_validations': 3}},
                                    validation_blocks=2)


if __name__ == '__main__':
    launch_example()
