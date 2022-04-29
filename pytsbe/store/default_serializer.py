from typing import Union

from pytsbe.data.forecast_output import ForecastResults


class DefaultLibrarySerializer:

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

    def save_information(self, ts_id: Union[int, str], forecast: ForecastResults):
        self._predictions_to_csv(ts_id, forecast)
        self._store_additional_info(ts_id, forecast)

    def _predictions_to_csv(self, ts_id: Union[int, str], forecast: ForecastResults):
        """ Save predictions ad pandas csv file """
        raise NotImplementedError()

    def _store_additional_info(self, ts_id: Union[int, str], forecast: ForecastResults):
        """ Save additional information after model fitting """
        return None
