from pytsbe.main import TimeSeriesLauncher


def launch_example():
    """
    Example how to launch benchmark with several libraries with different parameters.

    working_dir - directory for saving results
    launches - number of launches for averaging
    For more detailed info check documentation or docstring descriptions in classes below.
    """
    experimenter = TimeSeriesLauncher(working_dir='./example_launch',
                                      datasets=['FRED', 'SMART'],
                                      launches=2)
    experimenter.perform_experiment(libraries_to_compare=['FEDOT', 'repeat_last'],
                                    horizons=[10, 50],
                                    libraries_params={'FEDOT': {'preset': 'ts', 'timeout': 2,
                                                                'predefined_model': 'auto'}},
                                    validation_blocks=2,
                                    clip_border=500)


if __name__ == '__main__':
    launch_example()
