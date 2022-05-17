from pytsbe.report.walk import FolderWalker


class MetricsReport:
    """ A class for preparing summary tables with metrics from experiments """

    def __init__(self, working_dir: str):
        self.walker = FolderWalker(working_dir)

    def time_execution_table(self, batch_horizons: dict = None):
        pass

    def metric_table(self):
        pass
