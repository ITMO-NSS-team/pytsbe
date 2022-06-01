import json
import os


class FolderWalker:
    """
    Check folder with results. Walk through the folders and define paths to various files.
    If any values are not counted in one of the frameworks, they will be excluded in competitors.
    Thus, the class ensures consistency of results in the analysis.
    """

    def __init__(self, working_dir: str):
        self.working_dir = os.path.abspath(working_dir)

        path_to_config_json = os.path.join(self.working_dir, 'configuration.json')
        with open(path_to_config_json) as file:
            config_info = json.load(file)

        self.datasets = config_info['Datasets']
        self.launches = config_info['Launches']
        self.libraries = config_info['Libraries to compare']
        self.clip_border = config_info['Clip border']

        self.forecast_files = {}
        self.timeout_files = {}
        self.additional_files = {}
        for dataset in self.datasets:
            for launch in range(self.launches):
                for library in self.libraries:
                    launch_number = f'launch_{launch}'
                    case_id = f'{dataset}|{launch}|{library}'
                    validation_case_path = os.path.join(self.working_dir, dataset, launch_number, library)
                    all_forecasted_paths = self.find_files(validation_case_path,
                                                           search_pattern='forecast_vs_actual.csv')
                    self.forecast_files.update({case_id: all_forecasted_paths})

                    all_timeouts_paths = self.find_files(validation_case_path,
                                                         search_pattern='timeouts.json')
                    self.timeout_files.update({case_id: all_timeouts_paths})

                    all_additional_paths = self.find_additional_files(validation_case_path)
                    if all_additional_paths is not None:
                        self.additional_files.update({case_id: all_additional_paths})

        self.exclude_mismatched_results()

    def exclude_mismatched_results(self):
        """
        In some cases it is not possible to get results for some cases (dataset -
        launch number - library - time series - forecast horizon). So there is a
        need to exclude cases without calculations
        """
        for dataset in self.datasets:
            # First cycle - collect information
            dataset_execution_time = []
            dataset_forecast = []
            for launch in range(self.launches):
                for library in self.libraries:
                    case_id = f'{dataset}|{launch}|{library}'

                    ex_time_files = set(map(lambda x: os.path.basename(x), self.timeout_files[case_id]))
                    forecast_files = set(map(lambda x: os.path.basename(x), self.forecast_files[case_id]))

                    dataset_execution_time.append(ex_time_files)
                    dataset_forecast.append(forecast_files)

            # Find intersection for all cases
            dataset_execution_time = set.intersection(*dataset_execution_time)
            dataset_forecast = set.intersection(*dataset_forecast)

            # Second cycle - update info
            for launch in range(self.launches):
                for library in self.libraries:
                    case_id = f'{dataset}|{launch}|{library}'
                    ex_time_file = self.timeout_files[case_id][0]
                    current_path = os.path.dirname(ex_time_file)

                    upd_time_paths = add_path_to_files(current_path, dataset_execution_time)
                    upd_forecasts = add_path_to_files(current_path, dataset_forecast)
                    self.timeout_files.update({case_id: upd_time_paths})
                    self.forecast_files.update({case_id: upd_forecasts})

    @staticmethod
    def find_files(folder_with_files: str, search_pattern: str):
        """ Find all files in the folder and return full paths """
        files = os.listdir(folder_with_files)
        files.sort()
        all_paths = []
        for file in files:
            if search_pattern in file:
                all_paths.append(os.path.join(folder_with_files, file))

        return all_paths

    @staticmethod
    def find_additional_files(folder_with_files: str):
        """ Search for unusual files in saved folder - additional info """
        files = os.listdir(folder_with_files)
        files.sort()
        extra_paths = []
        for file in files:
            if 'timeouts.json' not in file and 'forecast_vs_actual.csv' not in file:
                extra_paths.append(os.path.join(folder_with_files, file))

        if len(extra_paths) == 0:
            return None

        return extra_paths


def add_path_to_files(current_path: str, files: set):
    """ In set with file names for each file add folder path """
    updated_data = []
    for file in files:
        updated_data.append(os.path.join(current_path, file))
    updated_data.sort()

    return updated_data
