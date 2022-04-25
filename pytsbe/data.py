from typing import List, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np

from pytsbe.paths import get_path_for_dataset


@dataclass
class TimeSeriesDatasets:
    """ Class for time series datasets preparing """
    # Time series for experiments
    time_series: Union[List[np.array], List[pd.DataFrame]]
    # Labels for every dataset, list must matching
    labels: List[str]

    @staticmethod
    def configure_dataset_from_path(dataset_name: str, clip_border: int = None):
        """ Prepare time series based on dataset name

        :param dataset_name: name of dataset to parse
        :param clip_border: is there a need to clip time series (if None - there is no cropping)
        """
        format_by_dataset_name = {'FRED': TimeSeriesDatasets.setup_from_long_format,
                                  'SMART': TimeSeriesDatasets.setup_from_long_format,
                                  'TEP': TimeSeriesDatasets.setup_from_long_format}

        # Get appropriate method and path to dataset
        prepare_method = format_by_dataset_name.get(dataset_name)
        dataset_path = get_path_for_dataset(dataset_name)
        val_set = prepare_method(path=dataset_path, clip_to=clip_border)
        return val_set

    @staticmethod
    def setup_from_long_format(path: str, series_id: str = 'series_id', clip_to: int = None):
        """ Load data from csv file with long format

        Structure can look like this
        datetime | value | series_id
           ---   |  ---  |    0
           ---   |  ---  |    0
           ---   |  ---  |    1

        :param path: path to csv file
        :param series_id: name of column with series labels
        :param clip_to: clip time series to desired length
        """
        df = pd.read_csv(path)

        datasets = []
        labels = []
        for i in df[series_id].unique():
            # Prepare pandas DataFrame for each time series
            local_df = df[df[series_id] == i]

            # Clip time series if its necessary
            if clip_to is not None:
                local_df = local_df.tail(clip_to)
            time_series = local_df[['datetime', 'value']]

            datasets.append(time_series)
            labels.append(i)

        return TimeSeriesDatasets(time_series=datasets, labels=labels)

    @staticmethod
    def setup_from_wide_format(path: str, clip_to: int = None):
        """ Load data from csv file with wide format

        Structure can look like this
        datetime | series_0 | series_1
           ---   |   ---    |   ---
           ---   |   ---    |   ---
           ---   |   ---    |   ---
        :param path: path to csv file
        :param clip_to: clip time series to desired length
        """

        df = pd.read_csv(path)
        datasets = []
        labels = []
        for i in df.columns:
            if i != 'datetime':
                time_series = df[['datetime', i]]
                # Clip time series if its necessary
                if clip_to is not None:
                    time_series = time_series.tail(clip_to)

                time_series = time_series.rename(columns={i: 'value'})

                datasets.append(time_series)
                labels.append(i)

        return TimeSeriesDatasets(time_series=datasets, labels=labels)
