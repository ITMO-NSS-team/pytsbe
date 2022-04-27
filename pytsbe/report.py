import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt


class Reporter:
    """ Class for preparing reports and visualisations. Also class allow to
    calculate new metrics for different forecast horizons based on saved predictions
    """
    def __init__(self, working_dir, datasets_info, launches: int = 1):
        self.working_dir = working_dir
        self.datasets_info = datasets_info
        self.launches = launches

    def prepare_ts_report(self, metrics: List[str], validation_blocks: int = None,
                          save_file: str = 'comparison.csv',
                          batches: dict = None):
        """ Create report with calculated metrics """

        libraries = self.get_processed_libraries()

        metric_final = []
        datasets_final = []
        libs_final = []
        min_max_batch = []
        metric_values_final = []
        metric_std_final = []
        for dataset_name, dataset_info in self.datasets_info.items():
            dataset_dir = os.path.join(self.working_dir, dataset_name)

            for lib in libraries:
                dataset_lib_path = os.path.join(dataset_dir, lib)
                dataset_lib_path = os.path.abspath(dataset_lib_path)
                result_batches = ts_calculate_metric_dataset_and_lib(metrics, dataset_lib_path,
                                                                     validation_blocks, batches)

                # For every metric
                for metric in metrics:
                    metric_values = result_batches[metric]

                    if batches is None:
                        # Process all horizons
                        all_results = metric_values['all']
                        mean_all = np.mean(all_results).round(2)
                        std_all = np.std(all_results).round(2)

                        metric_final.append(metric)
                        datasets_final.append(dataset_name)
                        libs_final.append(lib)
                        metric_values_final.append(mean_all)
                        metric_std_final.append(std_all)
                    else:
                        # Divide results on min (short) and max (long) horizons
                        min_results = metric_values['min']
                        max_results = metric_values['max']

                        mean_min = np.mean(min_results).round(2)
                        std_min = np.std(min_results).round(2)

                        mean_max = np.mean(max_results).round(2)
                        std_max = np.std(max_results).round(2)

                        metric_final.extend([metric, metric])
                        datasets_final.extend([dataset_name, dataset_name])
                        libs_final.extend([lib, lib])
                        metric_values_final.extend([mean_min, mean_max])
                        metric_std_final.extend([std_min, std_max])
                        min_max_batch.extend([str(batches['min']), str(batches['max'])])

        if batches is None:
            df = pd.DataFrame({'dataset': datasets_final, 'libraries': libs_final,
                               'metric': metric_final, 'metric value': metric_values_final,
                               'metric std': metric_std_final})
        else:
            df = pd.DataFrame({'dataset': datasets_final, 'libraries': libs_final,
                               'horizons': min_max_batch, 'metric': metric_final,
                               'metric value': metric_values_final, 'metric std': metric_std_final})
        df.to_csv(save_file, index=False)

    def show_ts_catplot(self, metric: str = 'SMAPE', validation_blocks: int = None, batches: dict = None):
        """ Function for visualizing results in the form of graphs (catplot) based on the seaborn library

        :param metric: name of metric to calculate
        :param validation_blocks: blocks for validation for in-sample forecasting
        :param batches: defines is it needed to calculate metric by horizons
        """
        libraries = self.get_processed_libraries()
        metric_vals = []
        horizons = []
        datasets = []
        libs = []
        # For every dataset
        for dataset_name, dataset_info in self.datasets_info.items():
            dataset_dir = os.path.join(self.working_dir, dataset_name)

            # For every library
            for lib in libraries:
                dataset_lib_path = os.path.join(dataset_dir, lib)
                dataset_lib_path = os.path.abspath(dataset_lib_path)
                result_batches = ts_calculate_metric_dataset_and_lib([metric], dataset_lib_path,
                                                                     validation_blocks, batches)
                metric_values = result_batches[metric]

                if batches is None:
                    # Process all horizons
                    all_results = metric_values['all']

                    metric_vals.extend(all_results)
                    datasets.extend([dataset_name] * len(all_results))
                    libs.extend([lib] * len(all_results))
                else:
                    metric_values = result_batches[metric]
                    min_results = metric_values['min']
                    max_results = metric_values['max']

                    # Update info about metrics for particular forecast horizons
                    metric_vals.extend(min_results)
                    horizons.extend(['short'] * len(min_results))

                    metric_vals.extend(max_results)
                    horizons.extend(['long'] * len(max_results))

                    full_len = len(min_results) + len(max_results)
                    datasets.extend([dataset_name] * full_len)
                    libs.extend([lib] * full_len)

        if batches is None:
            vis_df = pd.DataFrame({'dataset': datasets, 'library': libs,
                                   metric: metric_vals})
            with sns.axes_style("darkgrid"):
                sns.catplot(x='dataset', y=metric,
                            hue='library', data=vis_df, kind="strip",
                            dodge=True, height=4, aspect=.7)
                plt.show()

        else:
            vis_df = pd.DataFrame({'dataset': datasets, 'library': libs,
                                   'horizons': horizons, metric: metric_vals})

            with sns.axes_style("darkgrid"):
                sns.catplot(x='horizons', y=metric,
                            hue='library', col='dataset',
                            data=vis_df, kind="strip", dodge=True,
                            height=4, aspect=.7)
                plt.show()

    def show_ts_boxplot(self, metric: str = 'SMAPE', validation_blocks: int = None):
        """ Plot boxplots for time series forecasting task """
        libraries = self.get_processed_libraries()

        metric_vals = []
        datasets = []
        libs = []
        # For every dataset
        for dataset_name, dataset_info in self.datasets_info.items():
            dataset_dir = os.path.join(self.working_dir, dataset_name)

            # For every library
            for lib in libraries:
                dataset_lib_path = os.path.join(dataset_dir, lib)
                dataset_lib_path = os.path.abspath(dataset_lib_path)
                result_batches = ts_calculate_metric_dataset_and_lib(metrics=[metric],
                                                                     dataset_lib_path=dataset_lib_path,
                                                                     validation_blocks=validation_blocks)
                metric_values = result_batches[metric]
                all_results = metric_values['all']

                metric_vals.extend(all_results)
                datasets.extend([dataset_name] * len(all_results))
                libs.extend([lib] * len(all_results))

        vis_df = pd.DataFrame({'dataset': datasets, 'library': libs, metric: metric_vals})
        with sns.axes_style("darkgrid"):
            sns.boxplot(x='dataset', y=metric, width=0.5,
                        hue='library', data=vis_df, whis=np.inf)
            plt.show()

    def get_processed_libraries(self) -> np.array:
        """ Search for libraries, which were compared """
        libraries = []
        for dataset_name, dataset_info in self.datasets_info.items():
            dataset_dir = os.path.join(self.working_dir, dataset_name)

            libraries.extend(os.listdir(dataset_dir))

        unique_libraries = np.unique(np.array(libraries))
        return unique_libraries


def ts_calculate_metric_dataset_and_lib(metrics: list, dataset_lib_path: str,
                                        validation_blocks: int, batches: dict = None) -> dict:
    """ For considering path (particular library and particular dataset) calculates metrics

    :param metrics: list with names of metrics
    :param dataset_lib_path: path looking as "dataset_name/library_name"
    :param validation_blocks: number of blocks for in-sample validation
    :param batches: dictionary with forecast horizons for "mini"-batch and "max"-batch
    """
    metrics_by_name = {'MAE': {'metric': mean_absolute_error},
                       'RMSE': {'metric': mean_squared_error,
                                'params': {'squared': False}},
                       'SMAPE': {'metric': smape},
                       'R2': {'metric': r2_score}}

    launch_folders = os.listdir(dataset_lib_path)

    metric_results = {}
    for metric in metrics:
        min_batch_metrics = []
        max_batch_metrics = []
        all_batch_metrics = []

        metric_dict = metrics_by_name[metric]
        metric_func = metric_dict['metric']

        # For every launch of framework on the considering dataset
        for launch in launch_folders:
            forecast_path = os.path.join(dataset_lib_path, launch, 'forecasts')

            forecast_files = os.listdir(forecast_path)

            # For each forecast horizon for each time series
            for forecast_file in forecast_files:
                forecast_df = pd.read_csv(os.path.join(forecast_path, forecast_file))

                splitted = forecast_file.split('_')
                forecast_len = int(splitted[-1].split('.')[0])

                if validation_blocks is None:
                    horizon = forecast_len
                else:
                    horizon = forecast_len * validation_blocks

                # Take only validation part
                last_rows_df = forecast_df.tail(horizon)

                actual = np.array(last_rows_df['value'])
                predicted = np.array(last_rows_df['predicted'])

                if metric_dict.get('params') is None:
                    value = metric_func(actual, predicted)
                else:
                    value = metric_func(actual, predicted, **metric_dict['params'])

                if batches is None:
                    # There is no need to split into horizons - return all metrics
                    all_batch_metrics.append(value)
                else:
                    if forecast_len in batches['min']:
                        min_batch_metrics.append(value)
                    elif forecast_len in batches['max']:
                        max_batch_metrics.append(value)

        if batches is None:
            # Return all metrics
            all_batch_metrics = np.array(all_batch_metrics)

            metric_results.update({metric: {'all': all_batch_metrics}})
        else:
            min_batch_metrics = np.array(min_batch_metrics)
            max_batch_metrics = np.array(max_batch_metrics)

            metric_results.update({metric: {'min': min_batch_metrics,
                                            'max': max_batch_metrics}})
    return metric_results


def smape(y_true: np.array, y_pred: np.array) -> float:
    """ Symmetric mean absolute percentage error """

    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    result = numerator / denominator
    result[np.isnan(result)] = 0.0
    return float(np.mean(100 * result))
