from pytsbe.report.visualisation import Visualizer


def prepare_plots():
    """ Demonstration of how to use Visualizer to display plots """
    plots_creator = Visualizer(working_dir='./example_launch',
                               folder_for_plots=None)
    # The graphs show how long it takes to learn the models
    plots_creator.execution_time_comparison()

    # Visualize information about metrics for different cases
    plots_creator.metrics_comparison(metrics=['SMAPE'])

    # For each time series plot competitors libraries
    plots_creator.compare_forecasts()

    # Visualize one library but for different launches
    plots_creator.compare_launches(library='AutoTS')


if __name__ == '__main__':
    prepare_plots()
