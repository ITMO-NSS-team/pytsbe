import torch

from pytsbe.benchmark import BenchmarkUnivariate
from pytsbe.paths import get_project_path

seasons = ['Daily', 'Monthly', 'Quarterly', 'Weekly', 'Yearly']


def start_benchmark():
    """ Launch benchmark with desired configuration """
    print(f'CUDA is available: {torch.cuda.is_available()}')
    print(get_project_path())
    for season in seasons:
        pytsbe = BenchmarkUnivariate(working_dir=f'{get_project_path()}/experiments/univariate/benchmark_results_{season}',
                                     config_path=f'{get_project_path()}/experiments/univariate/configuration{season}.yaml')
        pytsbe.run()


if __name__ == '__main__':
    start_benchmark()
