import json
import os


class FolderWalker:
    """
    Check folder with results. Walk through the folders
    and define paths to various files.
    """

    def __init__(self, working_dir: str):
        self.working_dir = os.path.abspath(working_dir)

        path_to_config_json = os.path.join(self.working_dir, 'configuration.json')
        with open(path_to_config_json) as file:
            config_info = json.load(file)

        self.datasets = config_info['Datasets']
        self.launches = config_info['Launches']
        self.libraries = config_info['Libraries to compare']

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

    @staticmethod
    def find_files(folder_with_files: str, search_pattern: str):
        """ Find all files in the folder and return full paths """
        files = os.listdir(folder_with_files)
        all_paths = []
        for file in files:
            if search_pattern in file:
                all_paths.append(os.path.join(folder_with_files, file))

        return all_paths

    @staticmethod
    def find_additional_files(folder_with_files: str):
        """ Search for unusual files in saved folder - additional info """
        files = os.listdir(folder_with_files)
        extra_paths = []
        for file in files:
            if 'timeouts.json' not in file and 'forecast_vs_actual.csv' not in file:
                extra_paths.append(os.path.join(folder_with_files, file))

        if len(extra_paths) == 0:
            return None

        return extra_paths
