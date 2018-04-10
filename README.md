# MBS-PBS 10% dataset utilities
Python utilities to explore and analyze the Public Release of Linkable 10% sample of Medicare Benefits Scheme (Medicare) and Pharmaceutical Benefits Scheme (PBS) Data.

The main contribution of this project is a bidirectional timestamp-guided attention model that can detect which diabetic patient currently on metformin are likely to be prescribed witha different diabetes control drug in the next future.

**Note**: the MBS-PBS 10% is no longer publicly available (see [here](http://www.pbs.gov.au/info/news/2016/08/public-release-of-linkable-10-percent-mbs-and-pbs-data)) and it is not part of this repository.

## Installation

**Note**: this project requires Python 2.7. It does not currently work with Python 3.

To use this project for MBS-PBS 10% dataset exploration and analysis, you need to install two sets of tools:

- the core `mbspbs10pc` Package and its dependencies
- the renderer for the frontend you wish to use (i.e. `Jupyter Notebook` or `JupyterLab`)

If you have cloned the repository, run the following command from the root of the repository:

`$ cd mbspbs10pc`
`$ python setup.py install`

If you do not wish to clone the repository, you can install using:

`$ pip install git+https://github.com/samuelefiorini/mbspbs10pc.git`

## More info

For more info see: [pbs.gov.au](http://www.pbs.gov.au/info/news/2016/08/public-release-of-linkable-10-percent-mbs-and-pbs-data)
