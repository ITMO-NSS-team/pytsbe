from pytsbe.main import TimeSeriesLauncher


def multivariate_launch_example():
    """
    Example how to launch benchmark with several libraries with different
    parameters for multivariate time series forecasting

    For more detailed info check documentation or docstring descriptions in classes below.
    Important! The parameter 'predefined_model' for FEDOT framework does not launch the AutoML process.
    It should be removed to use AutoML.
    """
    experimenter = TimeSeriesLauncher(working_dir='./example_multivariate_launch',
                                      datasets=['SSH'],
                                      launches=2)
    experimenter.perform_experiment(libraries_to_compare=['FEDOT'],
                                    horizons=[20],
                                    libraries_params={'FEDOT': {'predefined_model': 'auto'}},
                                    validation_blocks=2,
                                    clip_border=400)


if __name__ == '__main__':
    multivariate_launch_example()
