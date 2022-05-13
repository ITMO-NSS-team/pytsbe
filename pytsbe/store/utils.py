import os


def create_folder(save_path):
    """ Create folder recursively """
    save_path = os.path.abspath(save_path)
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)
