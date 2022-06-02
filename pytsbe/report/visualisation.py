import os
from typing import Union

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pytsbe.data.data import TimeSeriesDatasets, MultivariateTimeSeriesDatasets, dataclass_by_name
from pytsbe.report.preparers.utils import get_label_and_horizon
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

    def compare_forecasts(self):
        """
        Plot comparison of forecasted values from different libraries with actual.
        Perform comparison for each case (dataset - launch - time series - forecast horizon)
        """
        libraries = self.metrics_processor.walker.libraries
        for dataset_name in self.metrics_processor.walker.datasets:
            # Prepare dataset
            clip_border = self.metrics_processor.walker.clip_border
            dataset_processor = dataclass_by_name[dataset_name]
            dataset = dataset_processor.configure_dataset_from_path(dataset_name=dataset_name,
                                                                    clip_border=clip_border)

            for launch_number in range(self.metrics_processor.walker.launches):
                case_ids = list(map(lambda x: f'{dataset_name}|{launch_number}|{x}', libraries))
                number_of_results = len(self.metrics_processor.walker.forecast_files[case_ids[0]])

                for number in range(number_of_results):
                    self._library_comparison_plot(dataset=dataset, dataset_name=dataset_name,
                                                  case_ids=case_ids, number=number, libraries=libraries,
                                                  launch_number=launch_number)

    def compare_launches(self, library: str):
        """ Plot forecasts and actual values for different launches for desired library """
        if library not in self.metrics_processor.walker.libraries:
            raise ValueError(f'Library "{library}" not in the list of results')

        for dataset_name in self.metrics_processor.walker.datasets:
            clip_border = self.metrics_processor.walker.clip_border
            dataset_processor = dataclass_by_name[dataset_name]
            dataset = dataset_processor.configure_dataset_from_path(dataset_name=dataset_name,
                                                                    clip_border=clip_border)
            first_case_id = list(self.metrics_processor.walker.forecast_files.keys())[0]
            number_of_results = len(self.metrics_processor.walker.forecast_files[first_case_id])

            for number in range(number_of_results):
                self._launch_comparison_plot(dataset, dataset_name, number, library)

    def _library_comparison_plot(self, dataset: Union[TimeSeriesDatasets, MultivariateTimeSeriesDatasets],
                                 dataset_name: str, case_ids: list, number: int, libraries: list, launch_number: int):
        """ Prepare table with values for visualization of competitive libraries
        and display (or save) plots where forecasts of different libraries are shown

        :param dataset: dataclass with time series
        :param dataset_name: current dataset name
        :param case_ids: cases for visualization
        :param number: number of time series in case for visualization
        :param libraries: libraries for comparison
        :param launch_number: current launch number
        """
        dataframe_for_comparison, plot_title = self._get_df_for_libraries_visualization(dataset, case_ids, number)
        dataframe_for_comparison['datetime'] = pd.to_datetime(dataframe_for_comparison['datetime'])

        train = dataframe_for_comparison[dataframe_for_comparison['Library'] == 'Actual values']
        test = dataframe_for_comparison[dataframe_for_comparison['Library'] != 'Actual values']
        plt.plot(train['datetime'], train['value'], label='Actual values')
        for library in libraries:
            library_test = test[test['Library'] == library]
            plt.plot(library_test['datetime'], library_test['predict'], label=library)

        first_index = test['datetime'].iloc[0]
        plt.plot([first_index, first_index], [min(train['value']), max(train['value'])], c='black',
                 alpha=0.5)
        plt.grid()
        plt.legend(fontsize=13)
        plt.xlabel('Datetime', fontsize=13)
        plt.ylabel('Parameter', fontsize=13)
        plt.title(plot_title, fontsize=13)
        if self.folder_for_plots is not None:
            # Save plot into folder
            name = f'Libraries - Dataset {dataset_name}, Launch {launch_number}, {plot_title}.png'
            fig_name = os.path.join(self.folder_for_plots, name)
            plt.savefig(fig_name)
            plt.close()
        else:
            plt.show()

    def _launch_comparison_plot(self, dataset: Union[TimeSeriesDatasets, MultivariateTimeSeriesDatasets],
                                dataset_name: str, number: int, library: str):
        """ Launch comparison for different launches

        :param dataset: dataclass with time series
        :param dataset_name: current dataset name
        :param number: number of time series in case for visualization
        :param library: name of library to show forecasts
        """
        current_ts_df, plot_title = self._get_df_for_launches_visualization(dataset=dataset, dataset_name=dataset_name,
                                                                            library=library, number=number)
        current_ts_df['datetime'] = pd.to_datetime(current_ts_df['datetime'])

        train = current_ts_df[current_ts_df['Launch number'] == 'source']
        test = current_ts_df[current_ts_df['Launch number'] != 'source']
        plt.plot(train['datetime'], train['value'], label='Actual values')
        for launch_number in range(self.metrics_processor.walker.launches):
            launch_test = test[test['Launch number'] == launch_number]
            plt.plot(launch_test['datetime'], launch_test['predict'], label=f'Launch number {launch_number}')

        first_index = test['datetime'].iloc[0]
        plt.plot([first_index, first_index], [min(train['value']), max(train['value'])], c='black',
                 alpha=0.5)
        plt.grid()
        plt.legend(fontsize=13)
        plt.xlabel('Datetime', fontsize=13)
        plt.ylabel('Parameter', fontsize=13)
        plt.title(plot_title, fontsize=13)
        if self.folder_for_plots is not None:
            # Save plot into folder
            name = f'Launches - Dataset {dataset_name}, {plot_title}.png'
            fig_name = os.path.join(self.folder_for_plots, name)
            plt.savefig(fig_name)
            plt.close()
        else:
            plt.show()

    def _get_df_for_libraries_visualization(self, dataset: Union[TimeSeriesDatasets, MultivariateTimeSeriesDatasets],
                                            case_ids: list, number: int):
        """ Create dataframe with predictions from different libraries for particular case

        :param dataset: dataset with time series which were used for model fitting
        :param case_ids: list with cases to check
        :param number: number of file in the case to take
        """
        plot_title = None
        dataframe_for_comparison = []
        for case_number, case_id in enumerate(case_ids):
            dataset_name, launch, library = case_id.split('|')
            # Get file with predictions and actual values
            forecasted_files = self.metrics_processor.walker.forecast_files[case_id]
            forecasted_files.sort()

            file_to_load = forecasted_files[number]
            if case_number == 0:
                ts_label, forecast_horizon = get_label_and_horizon(file_to_load, '_forecast_vs_actual.csv')
                # Add training sample of time series
                train_df = dataset.get_time_series_by_label(ts_label)
                train_df['Library'] = ['Actual values'] * len(train_df)
                dataframe_for_comparison.append(train_df)
                plot_title = f'Time series {ts_label}, Horizon {forecast_horizon}'

            df = pd.read_csv(file_to_load, parse_dates=['datetime'])
            df['Library'] = [library] * len(df)
            dataframe_for_comparison.append(df)
        dataframe_for_comparison = pd.concat(dataframe_for_comparison)

        return dataframe_for_comparison, plot_title

    def _get_df_for_launches_visualization(self, dataset: Union[TimeSeriesDatasets, MultivariateTimeSeriesDatasets],
                                           dataset_name: str, library: str, number: int):
        """ Create dataframe with predictions for one library from different launches for particular case

        :param dataset: dataset with time series which were used for model fitting
        :param dataset_name: name of dataset
        :param library: name of library
        :param number: number of file in the case to take
        """
        plot_title = None
        current_ts_df = []
        for launch_number in range(self.metrics_processor.walker.launches):
            case_id = f'{dataset_name}|{launch_number}|{library}'
            forecasted_files = self.metrics_processor.walker.forecast_files[case_id]
            forecasted_files.sort()
            df = pd.read_csv(forecasted_files[number], parse_dates=['datetime'])
            df['Launch number'] = [launch_number] * len(df)

            if launch_number == 0:
                ts_label, forecast_horizon = get_label_and_horizon(forecasted_files[number],
                                                                   '_forecast_vs_actual.csv')
                # Add training sample of time series
                train_df = dataset.get_time_series_by_label(ts_label)
                train_df['Launch number'] = ['source'] * len(train_df)
                current_ts_df.append(train_df)
                plot_title = f'Library {library}, Time series {ts_label}, Horizon {forecast_horizon}'

            current_ts_df.append(df)

        current_ts_df = pd.concat(current_ts_df)
        return current_ts_df, plot_title
