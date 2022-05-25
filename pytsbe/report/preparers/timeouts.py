import json
import pandas as pd

from pytsbe.report.preparers.utils import get_label_and_horizon
from pytsbe.report.walk import FolderWalker


def collect_timeouts_table(walker: FolderWalker) -> pd.DataFrame:
    """
    Create a table with all timeouts information from different launches.
    Goes through all the json files.

    :param walker: folder walker which contain information about directory structure
    """
    table = []
    for case_id, saved_paths in walker.timeout_files.items():
        dataset, launch, library = case_id.split('|')

        # Load dictionary with results
        for file_path in saved_paths:
            with open(file_path) as file:
                timeouts_data = json.load(file)

            # Get forecast horizon and time series name
            ts_label, forecast_horizon = get_label_and_horizon(file_path, '_timeouts.json')

            case_info = [dataset, launch, library, ts_label, forecast_horizon,
                         timeouts_data['fit_seconds'], timeouts_data['predict_seconds']]
            table.append(case_info)

    table = pd.DataFrame(table, columns=['Dataset', 'Launch', 'Library', 'Label', 'Horizon',
                                         'Fit, seconds', 'Predict, seconds'])
    return table
