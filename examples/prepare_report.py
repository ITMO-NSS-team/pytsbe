from pytsbe.report.report import MetricsReport


def prepare_tables_with_metric():
    """
    Demonstration of how to aggregate results and calculate metric for already
    calculated cases
    """
    metrics_processor = MetricsReport(working_dir='./example_launch')
    timeouts_table = metrics_processor.time_execution_table()
    print(timeouts_table)

    # TODO finish this example


if __name__ == '__main__':
    prepare_tables_with_metric()
