from typing import Union

from pytsbe.report.preparers.metrics import collect_metrics_table
from pytsbe.report.preparers.timeouts import collect_timeouts_table
from pytsbe.report.walk import FolderWalker


class MetricsReport:
    """ A class for preparing summary tables with metrics from experiments """

    def __init__(self, working_dir: str):
        self.walker = FolderWalker(working_dir)

    def time_execution_table(self, aggregation: Union[str, list] = None, agg_method: str = 'mean'):
        """
        Get dataframe with desired aggregation and information about timeouts

        :param aggregation: name of column for aggregation or several columns.
        Possible variants: 'Dataset', 'Launch', 'Library', 'Label', 'Horizon'
        If None, return full column.
        """
        timeouts_table = collect_timeouts_table(self.walker)
        if aggregation is None:
            return timeouts_table

        aggregated = timeouts_table.groupby(by=aggregation).agg({'Fit, seconds': agg_method,
                                                                 'Predict, seconds': agg_method})
        aggregated = aggregated.reset_index()
        return aggregated

    def metric_table(self, metrics: Union[str, list], aggregation: Union[str, list] = None, agg_method: str = 'mean'):
        """ Prepare table with desired metrics """
        if isinstance(metrics, str):
            metrics = [metrics]

        metric_table = collect_metrics_table(self.walker, metrics)
        if aggregation is None:
            return metric_table

        aggregation_columns = dict.fromkeys(metrics, agg_method)
        aggregated = metric_table.groupby(by=aggregation).agg(aggregation_columns)
        aggregated = aggregated.reset_index()
        return aggregated
