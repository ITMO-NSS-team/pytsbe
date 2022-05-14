from contextlib import contextmanager
from typing import Union
import functools


class ExceptionHandler:
    """ Class for exception handling. Try to avoid random failures during launches """

    def __init__(self, ts_label: Union[int, str], horizon: int):
        self.ts_label = ts_label
        self.horizon = horizon

    @contextmanager
    def safe_process_launch(self):
        """ Wrap fit and predict process """
        # TODO refactor for several exceptions without contextmanager
        try:
            yield
        except Exception as ex:
            prefix = f'Exception "{str(ex)}" raised.'
            print(f'{prefix} Skip time series {self.ts_label} for horizon {self.horizon}.')
