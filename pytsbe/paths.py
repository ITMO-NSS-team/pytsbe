import os
from pathlib import Path


def get_project_path() -> str:
    return Path(__file__).parent.parent


def get_data_path() -> str:
    """ Path to csv file with time series """
    project_path = get_project_path()
    return os.path.join(project_path, 'data')


def get_path_for_dataset(dataset_name: str):
    path_by_dataset_name = {'FRED': os.path.join(get_data_path(), 'fred.csv'),
                            'SMART': os.path.join(get_data_path(), 'smart.csv'),
                            'TEP': os.path.join(get_data_path(), 'tep.csv'),
                            'SSH': os.path.join(get_data_path(), 'multivariate_ssh.csv')}
    return path_by_dataset_name[dataset_name]


def get_path_for_dummy_dataset():
    """ This dataset is used only for testing """
    project_path = get_project_path()
    return os.path.join(project_path, 'test', 'data', 'dummy.csv')
