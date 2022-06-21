from pytsbe.bencmark import BenchmarkUnivariate


def start_benchmark():
    """ Launch benchmark with desired configuration """
    pytsbe = BenchmarkUnivariate(working_dir='./benchmark')
    pytsbe.run()


if __name__ == '__main__':
    start_benchmark()
