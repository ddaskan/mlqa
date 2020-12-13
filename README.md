# MLQA 

<img src="https://raw.githubusercontent.com/ddaskan/mlqa/master/docs/_static/mlqa.png" align="right" width="120"/>

[![PyPI](https://img.shields.io/pypi/v/mlqa)](https://pypi.org/project/mlqa/)
[![tests](https://github.com/ddaskan/mlqa/workflows/tests/badge.svg)](https://github.com/ddaskan/mlqa/actions?query=workflow%3Atests)
[![Codecov](https://codecov.io/gh/ddaskan/mlqa/master.svg)](https://codecov.io/gh/ddaskan/mlqa/)
[![Documentation Status](https://readthedocs.org/projects/mlqa/badge/?version=latest)](https://mlqa.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mlqa)](https://pypi.org/project/mlqa/)
[![GitHub last commit](https://img.shields.io/github/last-commit/ddaskan/mlqa)](https://github.com/ddaskan/mlqa)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fddaskan%2Fmlqa)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fddaskan%2Fmlqa)

A Package to perform QA on data flows for Machine Learning.

## Introduction

MLQA is a Python package that is created to help data scientists, analysts and developers to perform quality assurance (i.e. QA) on [pandas dataframes](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and 1d arrays, especially for machine learning modeling data flows. It's designed to work with [logging](https://docs.python.org/3/library/logging.html) library to log and notify QA steps in a descriptive way. It includes stand alone functions (i.e. [checkers](mlqa/checkers.py)) for different QA activities and [DiffChecker](mlqa/identifiers.py) class for integrated QA capabilities on data.

## Installation

You can install MLQA with pip.

`pip install mlqa`

MLQA depends on Pandas and Numpy and works in Python 3.6+.

## Quickstart

[DiffChecker](mlqa/identifiers.py) is designed to perform QA on data flows for ML. You can easily save statistics from the origin data such as missing value rate, mean, min/max, percentile, outliers, etc., then to compare against the new data. This is especially important if you want to keep the prediction data under the same assumptions with the training data.

Below is a quick example on how it works, just initiate and save statistics from the input data.
```python
>>> from mlqa.identifiers import DiffChecker
>>> import pandas as pd
>>> dc = DiffChecker()
>>> dc.fit(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*50+[1]*50}))

```

Then, you can check on new data if it's okay for given criteria. Below, you can see some data that is very similar in column `mean_col` but increased NA count in column `na_col`. The default threshold is 0.5 which means it should be okay if NA rate is 50% more than the origin data. NA rate is 50% in the origin data so up to 75% (i.e. 50*(1+0.5)) should be okay. NA rate is 70% in the new data and, as expected, the QA passes.

```python
>>> dc.check(pd.DataFrame({'mean_col':[.99, 2.1]*50, 'na_col':[None]*70+[1]*30}))
True

```

See more examples at [Documentation/Quickstart](https://mlqa.readthedocs.io/en/latest/source/quickstart.html). You can also read the full documentation [here](https://mlqa.readthedocs.io/).

## Tests
Tests are written with [unittest](https://docs.python.org/3/library/unittest.html) and can be located in the [tests](tests/) folder. There are also some tests in docstring to be run by [doctest](https://docs.python.org/3/library/doctest.html).

## License
[MIT](https://github.com/ddaskan/mlqa/blob/master/LICENSE)
