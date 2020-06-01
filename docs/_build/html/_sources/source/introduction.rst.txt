Introduction
============

What is it?
-----------

MLQA is a Python package that is created to help data scientists, analysts and developers to perform quality assurance (i.e. QA) on `pandas dataframes <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ and 1d arrays, especially for machine learning modeling data flows. It's designed to work with `logging <https://docs.python.org/3/library/logging.html>`_ library to log and notify QA steps in a descriptive way. It includes stand alone functions (i.e. `checkers <checkers.html>`_) for different QA activities and `DiffChecker <identifiers.html#mlqa.identifiers.DiffChecker>`_ class for integrated QA capabilities on data.

Installation
------------

You can install MLQA with pip.

.. code-block::

	$ pip install mlqa

Or, from the source.

.. code-block::

	$ git clone https://github.com/ddaskan/mlqa.git
	$ python -m unittest discover
	$ python setup.py

MLQA depends on Pandas and Numpy.