import json
import os

from typing import Union

from pytsbe.data.forecast_output import ForecastResults


class DefaultLibrarySerializer:
    """ Class for predictions serialization and store basic launch information """
    def __init__(self, storage_paths: dict):
        self.storage_paths = storage_paths

        self.dataset_name = None
        self.launch_number = None
        self.library_name = None

    def set_configuration_params(self, dataset_name, launch_number, library_name):
        """ Store configuration parameters in attributes of current serializers """
        self.dataset_name = dataset_name
        self.launch_number = launch_number
        self.library_name = library_name

    def save_information(self, ts_id: Union[int, str], horizon: int, forecast: ForecastResults):
        """ Store predictions and additional information (if it is required)

        :param ts_id: current time series name or id
        :param horizon: forecast horizon length
        :param forecast: results of algorithm launch with predictions and actual values
        """
        self.store_basic_info(ts_id, horizon, forecast)
        self.store_additional_info(ts_id, horizon, forecast)

    def store_basic_info(self, ts_id: Union[int, str], horizon: int, forecast: ForecastResults):
        """ Save predictions as pandas csv file for each time series in the dataset """
        storage_path = self.storage_paths[f'{self.dataset_name}_{self.launch_number}_{self.library_name}']

        # Save prediction and actual values into csv file
        forecast.true_values['predict'] = forecast.predictions
        path_to_save = os.path.join(storage_path, f'{ts_id}_{horizon}_forecast_vs_actual.csv')
        forecast.true_values.to_csv(path_to_save, index=False)

        # Save information about time for launching
        path_to_save_json = os.path.join(storage_path, f'{ts_id}_{horizon}_timeouts.json')
        with open(path_to_save_json, 'w') as outfile:
            json.dump(forecast.timeouts, outfile)

    def store_additional_info(self, ts_id: Union[int, str], horizon: int, forecast: ForecastResults):
        """
        Save additional information after model fitting. If the method is
        not extended in the descendant classes, nothing is saved.
        """
        return None
