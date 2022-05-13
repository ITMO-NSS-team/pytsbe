import datetime
from contextlib import contextmanager


class BenchmarkTimer:
    """ Class for timing. All values in seconds """

    def __init__(self):
        self.fit_spend_time = datetime.timedelta(minutes=0)
        self.predict_spend_time = datetime.timedelta(minutes=0)

    @contextmanager
    def launch_fit(self):
        """ Wrap fit process with timer """
        starting_time_for_fit = datetime.datetime.now()
        yield
        self.fit_spend_time = datetime.datetime.now() - starting_time_for_fit

    @contextmanager
    def launch_predict(self):
        """ Wrap predict process with timer """
        starting_time_for_predict = datetime.datetime.now()
        yield
        self.predict_spend_time = datetime.datetime.now() - starting_time_for_predict

    @property
    def fit_time(self):
        return self.fit_spend_time.total_seconds()

    @property
    def predict_time(self):
        return self.predict_spend_time.total_seconds()
