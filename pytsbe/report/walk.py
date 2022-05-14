class FolderWalker:

    def __init__(self, working_dir: str):
        self.working_dir = working_dir

    def find_all_files(self):
        raise NotImplementedError()
