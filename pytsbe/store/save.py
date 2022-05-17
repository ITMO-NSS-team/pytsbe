import os

from typing import List

from pytsbe.store.automl_serializer import AutoMLSerializer
from pytsbe.store.default_serializer import DefaultLibrarySerializer
from pytsbe.store.fedot_serializer import FedotSerializer
from pytsbe.store.utils import create_folder


class Serialization:
    """ Class for creating folders and preparing particular serializers for libraries """
    lib_serializers_by_name = {'default': DefaultLibrarySerializer,
                               'FEDOT': FedotSerializer,
                               'TPOT': AutoMLSerializer,
                               'H2O': AutoMLSerializer}

    def __init__(self, working_dir: str):
        self.working_dir = os.path.abspath(working_dir)
        create_folder(self.working_dir)

        self.storage_paths = {}

    def get(self, library_name: str):
        """ Return appropriate library serializer """
        if library_name not in self.lib_serializers_by_name:
            # Use default limited serializer to store only predictions and timeouts
            serializer = self.lib_serializers_by_name.get('default')(self.storage_paths)
        else:
            serializer = self.lib_serializers_by_name.get(library_name)(self.storage_paths)
        return serializer

    def create_folders_for_results(self, datasets: List[str], launches: int,
                                   libraries_to_compare: List[str]):
        """ Generate new folders (if they do not exist) to store results in the future """
        # Demonstration of the structure clearly
        for dataset_folder in datasets:
            # First level - dataset folder
            dataset_folder_path = os.path.join(self.working_dir, dataset_folder)

            for launch_number in range(launches):
                # Second level - launch number
                launch_folder_path = os.path.join(dataset_folder_path, f'launch_{launch_number}')

                for library_folder in libraries_to_compare:
                    # Third level - library name
                    library_folder_path = os.path.join(launch_folder_path, library_folder)
                    create_folder(library_folder_path)

                    structure_key = f'{dataset_folder}_{launch_number}_{library_folder}'
                    self.storage_paths.update({structure_key: library_folder_path})
