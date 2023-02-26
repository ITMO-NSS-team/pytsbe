import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm

from pytsbe.data.data import TimeSeriesDatasets
from pytsbe.paths import get_path_for_dataset


class DataExplorer:
    """ Class for time series datasets exploration. The class allows to collect
    statistical data about datasets in a selected csv file.
    """

    def __init__(self, directory: str = None):
        if directory is None:
            self.dataset_names = ['FRED', 'TEP', 'SMART']
        else:
            files = os.listdir(os.path.abspath(directory))
            self.dataset_names = list(map(lambda x: x.split('.csv')[0], files))
            print(f'New datasets were found in the {directory}: {self.dataset_names}')

    def display_statistics(self):
        """ Display statistic for each dataset in the desired folder """
        info = []
        for dataset_name in self.dataset_names:
            # Prepare data in a form of appropriate dataclass
            dataset = TimeSeriesDatasets.configure_dataset_from_path(dataset_name=dataset_name,
                                                                     clip_border=None)
            number_of_time_series = len(dataset.time_series)
            min_len, mean_len, max_len = self.calculate_lengths(dataset)
            non_stationary_ratio = self.calculate_percentage_of_non_stationary_series(dataset)
            info.append([dataset_name, number_of_time_series, min_len, mean_len, max_len, non_stationary_ratio])

        info = pd.DataFrame(info, columns=['Dataset', 'Total number of time series', 'Minimal row length',
                                           'Average row length',
                                           'Maximal row length', 'Percentage of non-stationary time series'])
        pd.set_option('display.max_columns', None)
        print(info)

        return info

    def visualise_series(self):
        """ Display time series in the dataset """
        sns.set_theme(style="ticks")

        for dataset_name in self.dataset_names:
            dataset_path = get_path_for_dataset(dataset_name)
            df = pd.read_csv(dataset_path, parse_dates=['datetime'])

            series_in_the_dataset = list(df['label'].unique())
            fig, axs = plt.subplots(len(series_in_the_dataset), sharex=True, sharey=False)
            fig.suptitle(f'Dataset {dataset_name}')

            for i, label in enumerate(series_in_the_dataset):
                df_series = df[df['label'] == label]

                axs[i].plot(np.arange(len(df_series)), df_series['value'], c='red')
                axs[i].axes.yaxis.set_visible(False)

            plt.xlabel('Time index')
            plt.show()

    @staticmethod
    def calculate_lengths(dataset: TimeSeriesDatasets):
        """ Calculate mean length and minimum / maximum length of time series in the dataset """
        lens = [len(ts) for ts in dataset.time_series]
        return min(lens), np.mean(lens), max(lens)

    @staticmethod
    def calculate_percentage_of_non_stationary_series(dataset: TimeSeriesDatasets):
        """ The Dickey-Fuller test evaluates whether a series is stationary or not.
        Then for each dataset percentage of non-stationary time series is calculated
        """
        number_of_all_time_series = len(dataset.time_series)
        number_of_non_stationary_time_series = 0
        with tqdm(total=len(dataset.time_series)) as pbar:
            pbar.set_description(f'Non stationary series percent calculation')
            for ts in dataset.time_series:
                # Calculate statistic
                p_test = sm.tsa.stattools.adfuller(np.array(ts['value']))[1]

                if p_test > 0.05:
                    number_of_non_stationary_time_series += 1
                pbar.update(1)

        non_stationary_ratio = (number_of_non_stationary_time_series / number_of_all_time_series) * 100
        return round(non_stationary_ratio, 2)
