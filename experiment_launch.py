from pytsbe.main import TimeSeriesLauncher


def launch_example():
    """
    Example how to launch benchmark with several libraries with different parameters.

    working_dir - directory for saving results
    launches - number of launches for averaging
    For more detailed info check documentation or docstring descriptions in classes below.
    """
    experimenter = TimeSeriesLauncher(working_dir='./example_launch',
                                      datasets=['FRED', 'SMART', 'TEP'],
                                      launches=2)
    experimenter.perform_experiment(libraries_to_compare=['FEDOT'],
                                    horizons=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                    libraries_params={'FEDOT': {'preset': 'ts', 'timeout': 3, 'n_jobs': -1}},
                                    validation_blocks=2,
                                    clip_border=500)


if __name__ == '__main__':
    launch_example()
