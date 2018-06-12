# TANGLE
**TANGLE** is a timespan-guided neural attention mechanism that can be used to integrate sequences of Medicare services and the time span between them to provide interpretable patients representations, which can be further used for some target prediction task.

Predictive capabilities of **TANGLE** are demonstrated here with an application on the 10\% publicly available sample of deidentified, individual level, linked Medicare Benefits Schedule (MBS) and Pharmaceutical Benefits Scheme (PBS) electronic databases of Australia. Starting from sequences of MBS-items and timespans, **TANGLE** can predict which diabetic patient, currently on metformin only, is likely to be prescribed with a different type of diabetes controlling drug in the near future.

## Installation
**TANGLE** is developed using [keras](https://keras.io/) and currently requires Python 2.7.

To use **TANGLE** for your sequence classification task, you simply need to install the core `tangle` package and its dependencies.

If you have cloned the repository, run the following command from the root of the repository:

`$ cd tangle`

`$ python setup.py install`

If you do not wish to clone the repository, you can install using:

`$ pip install git+https://github.com/samuelefiorini/tangle.git`

## More info
- **Note**: the MBS-PBS 10% is no longer publicly available (see [here](http://www.pbs.gov.au/info/news/2016/08/public-release-of-linkable-10-percent-mbs-and-pbs-data)) and it is not part of this repository.
- For more info see: [pbs.gov.au](http://www.pbs.gov.au/info/news/2016/08/public-release-of-linkable-10-percent-mbs-and-pbs-data).
