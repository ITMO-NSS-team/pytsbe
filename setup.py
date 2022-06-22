import setuptools
from os.path import dirname, join


def read(*names, **kwargs):
    with open(join(dirname(__file__), *names), encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()


def extract_requirements(file_name):
    return [r for r in read(file_name).split('\n') if r and not r.startswith('#')]


def get_requirements():
    requirements = extract_requirements('requirements.txt')
    return requirements


setuptools.setup(
    name="pytsbe",
    version="0.1",
    author="NSS Lab",
    description="Python time series forecasting benchmark tool",
    keywords='time series, forecasting, benchmark',
    long_description_content_type="text/markdown",
    url="https://github.com/ITMO-NSS-team/pytsbe",
    python_requires='>=3.7',
    install_requires=get_requirements(),
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)