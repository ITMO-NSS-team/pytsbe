from typing import Union

import os

from pytsbe.data.forecast_output import ForecastResults
from pytsbe.store.default_serializer import DefaultLibrarySerializer
from pytsbe.store.utils import create_folder


class FedotSerializer(DefaultLibrarySerializer):
    """
    Serializer for FEDOT framework output. Can store predictions, serialize
    obtained pipelines, save composing history and pipelines pictures
    """

    def __init__(self, storage_paths: dict):
        super().__init__(storage_paths)

    def store_additional_info(self, ts_id: Union[int, str], horizon: int, forecast: ForecastResults):
        """
        Save additional information for FEDOT framework. Serialize history
        of optimization, obtained models and pictures of pipeline structure
        """
        storage_path = self.storage_paths[f'{self.dataset_name}_{self.launch_number}_{self.library_name}']
        folder_with_additional_info = os.path.join(storage_path, 'additional')
        create_folder(folder_with_additional_info)

        # Serialize model
        serialized_path = os.path.join(folder_with_additional_info, 'serialized_pipeline')
        if os.path.exists(serialized_path) is False:
            # This model has already been saved
            forecast.obtained_model.save(os.path.join(folder_with_additional_info, 'model'))
            self.rename_folder_with_model(folder_with_additional_info)

        # Save picture of model
        forecast.obtained_model.show(os.path.join(folder_with_additional_info, 'obtained_model.png'))

        # Save optimization history
        history = forecast.additional_info['fedot_api_object'].history
        if history is not None:
            # AutoML process with composing has been started and finished
            history.save(os.path.join(folder_with_additional_info, 'opt_history.json'))

    @staticmethod
    def rename_folder_with_model(folder_with_additional_info):
        folders = os.listdir(folder_with_additional_info)

        for folder in folders:
            if 'PM' in folder or 'AM' in folder:
                # Folder need to be renamed
                old_name = os.path.join(folder_with_additional_info, folder)
                new_name = os.path.join(folder_with_additional_info, 'serialized_pipeline')
                os.rename(old_name, new_name)
