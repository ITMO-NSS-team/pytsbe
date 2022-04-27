import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

from pytsbe.data.data import TimeSeriesDatasets


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

        info = pd.DataFrame(info, columns=['Dataset', 'Total number of time series', 'Average row length',
                                           'Minimal row length', 'Maximal row length', 'Percentage of non-stationary time series'])
        pd.set_option('display.max_columns', None)
        print(info)

        return info

    def visualise_series(self):
        """ Display time series in the dataset """
        raise NotImplementedError()

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
        for ts in dataset.time_series:
            # Calculate statistic
            p_test = sm.tsa.stattools.adfuller(np.array(ts['value']))[1]

            if p_test > 0.05:
                number_of_non_stationary_time_series += 1

        non_stationary_ratio = (number_of_non_stationary_time_series / number_of_all_time_series) * 100
        return round(non_stationary_ratio, 2)
