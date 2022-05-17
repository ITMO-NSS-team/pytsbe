import json
from typing import Union

import os

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.store.default_serializer import DefaultLibrarySerializer


class AutoMLSerializer(DefaultLibrarySerializer):
    """
    Serializer for AutoML frameworks output.
    Save information about window size for lagged transformation
    """

    def __init__(self, storage_paths: dict):
        super().__init__(storage_paths)

    def store_additional_info(self, ts_id: Union[int, str], horizon: int, forecast: ForecastResults):
        """ Save additional information for AutoML frameworks """
        storage_path = self.storage_paths[f'{self.dataset_name}_{self.launch_number}_{self.library_name}']
        lagged_window_size = forecast.additional_info['lagged_window_size']

        path_to_save_json = os.path.join(storage_path, f'{ts_id}_{horizon}_window_sizes.json')
        with open(path_to_save_json, 'w') as outfile:
            json.dump({'lagged window size': lagged_window_size}, outfile)
