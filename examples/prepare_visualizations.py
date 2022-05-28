from pytsbe.report.visualisation import Visualizer


def prepare_plots():
    """ Demonstration of how to use Visualizer to display plots """
    plots_creator = Visualizer(working_dir='./example_launch',
                               folder_for_plots=None)
    plots_creator.execution_time_comparison()

    plots_creator.metrics_comparison(metrics=['MAE', 'SMAPE'])


if __name__ == '__main__':
    prepare_plots()
