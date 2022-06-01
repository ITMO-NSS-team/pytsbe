from pytsbe.data.data import TimeSeriesDatasets, MultivariateTimeSeriesDatasets

dataclass_by_name = {'FRED': TimeSeriesDatasets,
                     'TEP': TimeSeriesDatasets,
                     'SMART': TimeSeriesDatasets,
                     'SSH': MultivariateTimeSeriesDatasets}
