from pytsbe.main import TimeSeriesLauncher


def multivariate_launch_example():
    """
    Example how to launch benchmark with several libraries with different
    parameters for multivariate time series forecasting

    For more detailed info check documentation or docstring descriptions in classes below
    """
    experimenter = TimeSeriesLauncher(working_dir='./example_multivariate_launch',
                                      datasets=['SSH'],
                                      launches=2)
    experimenter.perform_experiment(libraries_to_compare=['FEDOT'],
                                    horizons=[10, 50],
                                    libraries_params={'FEDOT': {'predefined_model': 'auto'}},
                                    validation_blocks=2,
                                    clip_border=500)


if __name__ == '__main__':
    multivariate_launch_example()
