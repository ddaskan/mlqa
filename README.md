# MLQA <img src="docs/_static/mlqa.png" align="right" width="120"/>

 A package to perform QA for Machine Learning Models.

 ## Introduction

 MLQA is a Python package that is created to help data scientists, analysts and developers to perform quality assurance (i.e. QA) on [pandas dataframes](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and 1d arrays, especially for machine learning modeling data flows. It's designed to work with [logging](https://docs.python.org/3/library/logging.html) library to log and notify QA steps in a descriptive way.

 ## Installation

 You can install MLQA with pip.
 
 `pip install mlqa`

 MLQA depends on Pandas and Numpy and works in Python 3.5+.

## Quickstart

You can easily initiate the object and fit a pd.DataFrame.
```python
>>> from mlqa.identifiers import DiffChecker
>>> import pandas as pd
>>> dc = DiffChecker()
>>> dc.fit(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*50+[1]*50}))
```

Then, you can check on new data if it's okay for given criteria. Below, you can see data with increased NA count in column `na_col`. The default threshold is 0.5 which means it should be okay if NA rate is 50% more than the fitted data. NA rate is 50% in the fitted data so up to 75% (i.e. 50*(1+0.5)) should be okay. NA rate is 70% in the new data and, as expected, the QA passes. 

```python
>>> dc.check(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*70+[1]*30}))
True
```

See more examples at [Documentation/Quickstart](http://www.doganaskan.com/mlqa/source/quickstart.html). You can also read the full documentation [here](http://www.doganaskan.com/mlqa/).

## Tests
Tests are written with [unittest](https://docs.python.org/3/library/unittest.html) and can be located in the [tests](tests/) folder. There are also some tests in docstring to be run by [doctest](https://docs.python.org/3/library/doctest.html).

## License
[MIT](LICENSE)
