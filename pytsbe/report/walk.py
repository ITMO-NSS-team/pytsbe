import json
import os
from os import walk


class FolderWalker:
    """ Check folder with results """

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
        for dataset in self.datasets:
            for launch in range(self.launches):
                for library in self.libraries:
                    launch_number = f'launch_{launch}'
                    validation_case_path = os.path.join(self.working_dir, dataset, launch_number, library)
                    all_forecasted_paths = self.find_files(validation_case_path,
                                                           search_pattern='forecast_vs_actual.csv')
                    self.forecast_files.update({f'{dataset}{launch}{library}': all_forecasted_paths})

                    all_timeouts_paths = self.find_files(validation_case_path,
                                                         search_pattern='timeouts.json')
                    self.timeout_files.update({f'{dataset}{launch}{library}': all_timeouts_paths})

    def find_files(self, folder_with_files, search_pattern: str):
        """ Find all files in the folder and return full paths """
        files = os.listdir(folder_with_files)
        all_paths = []
        for file in files:
            if search_pattern in file:
                all_paths.append(os.path.join(folder_with_files, file))

        return all_paths
