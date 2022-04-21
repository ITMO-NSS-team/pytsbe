import os
from pathlib import Path


def get_project_path() -> str:
    return Path(__file__).parent.parent


def get_data_path() -> str:
    """ Path to csv file with time series """
    project_path = get_project_path()
    return os.path.join(project_path, 'data')
