from pytsbe.benchmark import BenchmarkMultivariate


def start_benchmark():
    """ Launch benchmark with desired configuration """
    pytsbe = BenchmarkMultivariate(working_dir='./benchmark_results')
    pytsbe.run()


if __name__ == '__main__':
    start_benchmark()
