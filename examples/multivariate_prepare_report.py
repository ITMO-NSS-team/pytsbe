from pytsbe.report.report import MetricsReport


def prepare_tables_with_metric():
    """
    Demonstration of how to aggregate results and calculate metric for already
    calculated cases.
    Task: multivariate forecasting
    """
    metrics_processor = MetricsReport(working_dir='./example_multivariate_launch')
    timeouts_table = metrics_processor.time_execution_table(aggregation=['Library', 'Dataset'])
    print('Information about execution times:')
    print(timeouts_table)

    # Calculate and display information about metrics during validation
    metrics_table = metrics_processor.metric_table(metrics=['MAE', 'SMAPE'],
                                                   aggregation=['Library', 'Dataset', 'Horizon'])
    print('\nInformation about metrics:')
    print(metrics_table)


if __name__ == '__main__':
    prepare_tables_with_metric()
