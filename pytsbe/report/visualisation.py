from pytsbe.report.report import MetricsReport


class Visualizer:
    """ Class for creating various visualizations based on performed experiments """

    def __init__(self, working_dir: str):
        self.metrics_processor = MetricsReport(working_dir)

    def execution_time_comparison(self):
        timeouts_df = self.metrics_processor.time_execution_table()
        # TODO finish
