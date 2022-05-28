import os
from typing import Union

import seaborn as sns
import matplotlib.pyplot as plt

from pytsbe.report.report import MetricsReport
from pytsbe.store.utils import create_folder


class Visualizer:
    """
    Class for creating various visualizations based on experiment's results

    :param working_dir: directory where the results of experiments were saved
    :param folder_for_plots: directory to saved plots in it. If None, display
    plots without saving.
    """

    def __init__(self, working_dir: str, folder_for_plots: str = None):
        self.metrics_processor = MetricsReport(working_dir)
        self.folder_for_plots = None
        if folder_for_plots is not None:
            # Create folder for visualisations
            self.folder_for_plots = os.path.abspath(folder_for_plots)
            create_folder(self.folder_for_plots)

    def execution_time_comparison(self, palette: str = 'Set2'):
        """ Prepare stripplot for execution time """
        timeouts_df = self.metrics_processor.time_execution_table()
        horizons = list(timeouts_df['Horizon'].unique())
        horizons.sort()

        with sns.axes_style('darkgrid'):
            for y_values in ['Fit, seconds', 'Predict, seconds']:
                sns.catplot(x='Dataset', y=y_values, hue='Library',
                            col='Horizon', data=timeouts_df, palette=palette,
                            dodge=True, col_order=horizons)
                if self.folder_for_plots is not None:
                    # Save plot into folder
                    fig_name = os.path.join(self.folder_for_plots, f'{y_values}.png')
                    plt.savefig(fig_name)
                    plt.close()
                else:
                    plt.show()

    def metrics_comparison(self, metrics: Union[str, list], palette: str = 'Set3'):
        """ Prepare stripplots for metrics """
        if isinstance(metrics, str):
            metrics = [metrics]

        metric_table = self.metrics_processor.metric_table(metrics=metrics)
        horizons = list(metric_table['Horizon'].unique())
        horizons.sort()
        with sns.axes_style('darkgrid'):
            for y_values in metrics:
                sns.catplot(x='Dataset', y=y_values, hue='Library',
                            col='Horizon', data=metric_table, palette=palette,
                            dodge=True, col_order=horizons)
                if self.folder_for_plots is not None:
                    # Save plot into folder
                    fig_name = os.path.join(self.folder_for_plots, f'{y_values}.png')
                    plt.savefig(fig_name)
                    plt.close()
                else:
                    plt.show()
