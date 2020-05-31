Quickstart
==========

Here, you can see some quick examples on how to utilize the package. For more details, refer to `API Reference <../index.html#api-reference>`_.

DiffChecker Basics
------------------

`DiffChecker <identifiers.html#identifiers.DiffChecker>`_ is designed to perform QA in an integrated way on pd.DataFrame.

You can easily initiate the object and fit a pd.DataFrame.

.. code-block:: python

	>>> dc = DiffChecker()
	>>> dc.fit(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*50+[1]*50}))

Then, you can check on new data if it's okay for given criteria. Below, you can see data with increased NA count in column `na_col`. The default threshold is 0.5 which means it should be okay if NA rate is 50% more than fitted data. NA rate is 50% in the fitted data so up to 75% (i.e. 50*(1+0.5)) should be okay. NA rate is 70% in the new data and, as expected, the QA passes. 

.. code-block:: python

	>>> dc.check(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*70+[1]*30}))
	True

If you think the `threshold <identifiers.html#identifiers.DiffChecker.threshold>`_ is too loose, you can adjust as you wish with `set_threshold <identifiers.html#identifiers.DiffChecker.set_threshold>`_ method. And, now the same returns `False` indicating the QA has failed.

.. code-block:: python

	>>> dc.set_threshold(0.1)
	>>> dc.check(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*70+[1]*30}))
	False

DiffChecker Details
-------------------

As default, `DiffChecker <identifiers.html#identifiers.DiffChecker>`_ is initialized with `qa_level='loose'`. Different values can also be given.

.. code-block:: python

	>>> dc = DiffChecker()
	>>> dc.threshold
	0.5
	>>> dc = DiffChecker(qa_level='mid')
	>>> dc.threshold
	0.2
	>>> dc = DiffChecker(qa_level='strict')
	>>> dc.threshold
	0.1

To be more precise, you can set both `threshold <identifiers.html#identifiers.DiffChecker.threshold>`_ and `stats <identifiers.html#identifiers.DiffChecker.stats>`_ individually.

.. code-block:: python

    >>> dc = DiffChecker()
    >>> dc.set_threshold(0.2)
    >>> dc.set_stats(['mean', 'max', np.sum])
    >>> dc.fit(pd.DataFrame({'col1':[1, 2, 3, 4], 'col2':[1]*4}))
    >>> dc.check(pd.DataFrame({'col1':[1, 2, 3, 4], 'col2':[0]*4}))
    False
    >>> dc.check(pd.DataFrame({'col1':[1, 2.1, 3.2, 4.2], 'col2':[1.1]*4}))
    True

You can even be more detailed in `set_threshold <identifiers.html#identifiers.DiffChecker.set_threshold>`_.

.. code-block:: python

    >>> dc = DiffChecker()
    >>> dc.set_stats(['mean', 'max'])
    >>> dc.set_threshold(0.1) # to reset all thresholds
    >>> print(dc.threshold)
    0.1
    >>> dc.fit(pd.DataFrame({'col1':[1, 2, 3, 4], 'col2':[0]*4}))
    >>> dc.set_threshold({'col1':0.2, 'col2':0.1}) # to set in column level
    >>> print(dc.threshold_df)
          col1  col2
    mean   0.2   0.1
    max    0.2   0.1
    >>> dc.set_threshold({'col1':{'mean':0.1}}) # to set in column-stat level
    >>> print(dc.threshold_df)
          col1  col2
    mean   0.1   0.1
    max    0.2   0.1

You can also pickle the object to be used later with `to_pickle <identifiers.html#identifiers.DiffChecker.to_pickle>`_ method.

.. code-block:: python

    >>> dc1 = DiffChecker()
    >>> dc1.fit(pd.DataFrame({'col1':[1, 2, 3, 4], 'col2':[0]*4}))
    >>> dc1.to_pickle(path='DiffChecker.pkl')

Then, to load the same object later.

.. code-block:: python

    >>> import pickle
    >>> pkl_file = open('DiffChecker.pkl', 'rb')
    >>> dc2 = pickle.load(pkl_file)
    >>> pkl_file.close()

DiffChecker with Logging
------------------------

If you enable logging functionality, you can get detailed description of what column failed for which stat and why. You can even log `DiffChecker <identifiers.html#identifiers.DiffChecker>`_ steps.

Just initiate the class with `logger='<your-logger-name>.log'` argument.

.. code-block:: python

    >>> dc = DiffChecker(logger='mylog.log')
    >>> dc.fit(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*50+[1]*50}))
    >>> dc.set_threshold(0.1)
    >>> dc.check(pd.DataFrame({'mean_col':[1, 1.5]*50, 'na_col':[None]*70+[1]*30}))
    False

If you open `mylog.log`, you'll see something like below.

.. code-block::

	WARNING|2020-05-31 15:56:48,146|mean value (i.e. 1.25) is not in the range of [1.35, 1.65] for mean_col
	WARNING|2020-05-31 15:56:48,147|na_rate value (i.e. 0.7) is not in the range of [0.45, 0.55] for na_col

If you initiate the class with also `log_info=True` argument, then the other class steps (e.g. `set_threshold <identifiers.html#identifiers.DiffChecker.set_threshold>`_, `check <identifiers.html#identifiers.DiffChecker.check>`_) would be logged, too.

Checkers with Logging
---------------------

There are also `checkers <checkers.html>`_ to provide other kind of QA functionalities such as `outliers detection <checkers.html#checkers.qa_outliers>`_, `pd.DataFrame comparison <checkers.html#checkers.qa_df_set>`_ or some `categorical value QA <checkers.html#checkers.qa_category_distribution_on_value>`_. You can use these individually or combining with `DiffChecker <identifiers.html#identifiers.DiffChecker>`_'s logger.




