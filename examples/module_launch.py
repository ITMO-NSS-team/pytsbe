from pytsbe.main import TimeSeriesLauncher


def launch_example():
    """
    Example how to launch benchmark with several libraries with different parameters.

    working_dir - directory for saving results
    datasets_info - dictionary with information about datasets
    launches - number of launches for averaging
    For more detailed info check documentation or docstring descriptions in classes below.
    """
    experimenter = TimeSeriesLauncher(working_dir='./example_launch',
                                      datasets=['FRED', 'TEP', 'SMART'],
                                      launches=5)
    experimenter.perform_experiment(libraries_to_compare=['ets', 'FEDOT', 'AutoTS'],
                                    horizons=[10],
                                    libraries_params={'H2O': {'timeout': 2, 'max_models': 10},
                                                      'TPOT': {'timeout': 2, 'generations': 50, 'population_size': 16},
                                                      'FEDOT': {'preset': 'ts', 'timeout': 2, 'predefined_model': 'auto'},
                                                      'AutoTS': {'frequency': 'infer', 'prediction_interval': 0.9,
                                                                 'ensemble': 'all', 'model_list': 'default',
                                                                 'max_generations': 1, 'num_validations': 3}},
                                    validation_blocks=2,
                                    clip_border=500)


if __name__ == '__main__':
    launch_example()
