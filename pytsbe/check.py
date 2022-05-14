from typing import Union
import os


class FailedLaunchChecker:
    """
    Class for checking current result folder status.
    Allow to skip calculations for already finished cases.
    """

    def __init__(self, storage_paths: dict, dataset_name: str,
                 launch_number: int, library_name: str):
        self.storage_paths = storage_paths
        self.dataset_name = dataset_name
        self.launch_number = launch_number
        self.library_name = library_name

    def was_case_finished(self, ts_id: Union[int, str], horizon: int) -> bool:
        """
        Check if current case (Dataset - launch - library - time series - horizon)
        has already been finished or not.

        :param ts_id: id of time series in the dataset
        :param horizon: forecast horizon length
        """
        storage_path = self.storage_paths[f'{self.dataset_name}_{self.launch_number}_{self.library_name}']
        predictions_path = os.path.join(storage_path, f'{ts_id}_{horizon}_forecast_vs_actual.csv')
        timeouts_path = os.path.join(storage_path, f'{ts_id}_{horizon}_timeouts.json')

        is_predictions_exist = os.path.exists(predictions_path)
        is_timeouts_exist = os.path.exists(timeouts_path)
        is_case_finished = is_predictions_exist and is_timeouts_exist

        if is_case_finished:
            prefix = f'Case dataset {self.dataset_name} - launch {self.launch_number} - library {self.library_name}'
            print(f'{prefix} - time series {ts_id} - horizon {horizon} was finished. Skip claculations.')

        return is_case_finished
