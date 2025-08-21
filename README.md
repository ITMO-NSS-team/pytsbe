# pytsbe

---

![License](https://img.shields.io/github/license/ITMO-NSS-team/pytsbe?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

Pytsbe is a tool designed to help researchers and developers rigorously compare the performance of different time series forecasting methods. It provides access to various algorithms and datasets, simplifying the process of evaluating and selecting the best model for a given prediction task.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---
## Core features

1. **Time Series Forecasting Benchmark**: Provides a framework for benchmarking various time series forecasting algorithms and libraries, enabling comparative analysis of their performance.
2. **Univariate Time Series Support**: Supports the evaluation of forecasting models on single-variable time series data, including datasets like FRED, TEP, and SMART.
3. **Multivariate Time Series Support**: Enables benchmarking of algorithms designed for multi-variable time series forecasting, currently featuring the SSH dataset.
4. **Library Integration**: Integrates with popular time series libraries such as FEDOT, AutoTS, pmdarima, Prophet, H2O, and TPOT, allowing for easy comparison.
5. **Configurable Benchmarking**: Allows users to configure benchmarking experiments through YAML configuration files, specifying datasets, libraries, horizons, and validation blocks.
6. **Automated Reporting**: Generates reports summarizing the performance of different forecasting models, including metrics like SMAPE and execution times.

---

## Installation

**Prerequisites:** requires Python >=3.8

Install pytsbe using one of the following methods:

**Build from source:**

1. Clone the pytsbe repository:
```sh
git clone https://github.com/ITMO-NSS-team/pytsbe
```

2. Navigate to the project directory:
```sh
cd pytsbe
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```
## Getting Started

The `TimeSeriesLauncher` class is used to run the experiments.

### Initialization parameters

`working_dir` - directory for saving algorithm output. If the directory does not exist, it will be created

`datasets` - a list of dataset names.

`launches` - number of launches to perform.

### perform_experiment method parameters

`libraries_to_compare` - a list of libraries names.

`horizons` - a list of forecast horizons names

`libraries_params` - dictionary with parameters for libraries.

`validation_blocks` - validation blocks for in-sample forecasting. If null or 1 - simple validation is made.

`clip_border` - number of elements to remain in time series if there is a need to clip time series (if null - there is no cropping).

Usage example:

```python
from pytsbe.main import TimeSeriesLauncher

experimenter = TimeSeriesLauncher(working_dir='./output',
                                  datasets=['FRED', 'TEP', 'SMART'],
                                  launches=2)

experimenter.perform_experiment(libraries_to_compare=['FEDOT', 'AutoTS', 'pmdarima', 'repeat_last'],
                                horizons=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                libraries_params={'FEDOT': {'preset': 'ts', 'timeout': 2},
                                                  'AutoTS': {'frequency': 'infer', 'prediction_interval': 0.9,
                                                             'ensemble': 'all', 'model_list': 'default',
                                                             'max_generations': 1, 'num_validations': 3}},
                                validation_blocks=3,
                                clip_border=1000)
```

<img src="./images/features_1.png" width="970"/> 

<img src="./images/features_2.png" width="970"/> 

<img src="./images/features_3.png" width="970"/>

---

## Examples

Examples of how this should work and how it should be used are available [here](https://github.com/ITMO-NSS-team/pytsbe/tree/main/examples).

---

## Documentation

A detailed pytsbe description is available [here](https://github.com/ITMO-NSS-team/pytsbe/tree/main/docs).

---

## Contributing

- **[Report Issues](https://github.com/ITMO-NSS-team/pytsbe/issues)**: Submit bugs found or log feature requests for the project.

- **[Submit Pull Requests](https://github.com/ITMO-NSS-team/pytsbe/tree/main/.github/CONTRIBUTING.md)**: To learn more about making a contribution to pytsbe.

---

## License

This project is protected under the BSD 3-Clause "New" or "Revised" License. For more details, refer to the [LICENSE](https://github.com/ITMO-NSS-team/pytsbe/tree/main/LICENSE.md) file.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    ITMO-NSS-team (2022). pytsbe repository [Computer software]. https://github.com/ITMO-NSS-team/pytsbe

### BibTeX format:

    @misc{pytsbe,

        author = {ITMO-NSS-team},

        title = {pytsbe repository},

        year = {2022},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/ITMO-NSS-team/pytsbe.git}},

        url = {https://github.com/ITMO-NSS-team/pytsbe.git}

    }

---
