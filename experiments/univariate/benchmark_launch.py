import torch

from pytsbe.benchmark import BenchmarkUnivariate

seasons = ['Daily', 'Monthly', 'Quarterly', 'Weekly', 'Yearly']


def start_benchmark():
    """ Launch benchmark with desired configuration """
    print(torch.cuda.is_available())
    for season in seasons:
        pytsbe = BenchmarkUnivariate(working_dir=f'./benchmark_results_{season}',
                                     config_path=f'configuration{season}.yaml')
        pytsbe.run()


if __name__ == '__main__':
    start_benchmark()
