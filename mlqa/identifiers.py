'''
This module is for DiffChecker class.
'''
import sys
import os
import logging
from importlib import reload
import pickle
import pandas as pd
import numpy as np
sys.path.append('../')
from mlqa import checkers as ch

class DiffChecker():
    '''Integrated QA performer on pd.DataFrame with logging functionality.

    It only works in numerical columns.

    Args:
        qa_level (str): quick set for QA level, can be one of ['loose', 'mid', 'strict']
        logger (str or logging.Logger): 'print' for print only, every other
            str creates a file for logging. using external logging.Logger object
            is highly recommended, i.e. logger=<mylogger>.
        qa_log_level (int): qa message logging level
        log_info (bool): `True` if method calls or arguments also need to be
            logged

    Example:
        Basic usage:

        >>> dc = DiffChecker()
        >>> dc.fit(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*50+[1]*50}))
        >>> dc.check(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*70+[1]*30}))
        True
        >>> dc.set_threshold(0.1)
        >>> dc.check(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*70+[1]*30}))
        False

        Quick set for `qa_level`:

        >>> dc = DiffChecker()
        >>> dc.threshold
        0.5
        >>> dc = DiffChecker(qa_level='mid')
        >>> dc.threshold
        0.2
        >>> dc = DiffChecker(qa_level='strict')
        >>> dc.threshold
        0.1

        Logger can also be initiated:

        >>> dc = DiffChecker(logger='mylog.log')
        >>> dc.fit(pd.DataFrame({'mean_col':[1, 2]*50, 'na_col':[None]*50+[1]*50}))
        >>> dc.set_threshold(0.1)
        >>> dc.check(pd.DataFrame({'mean_col':[1, 1.5]*50, 'na_col':[None]*70+[1]*30}))
        False
        >>> os.remove('mylog.log')

    '''
    stats = []
    threshold = 0.0
    threshold_df = pd.DataFrame()
    df_fit_stats = pd.DataFrame()

    def __init__(
            self,
            qa_level='loose',
            logger=None,
            qa_log_level=None,
            log_info=False
    ):

        # Class logger reloads logging module in each call not to create
        # conflict, this is okay as long as this is the only logger in the
        # environment. Having external logger is highly recommended in all
        # other cases.
        if logger == 'print':
            logging.shutdown()
            reload(logging)
            logging.basicConfig(
                format='%(asctime)-15s %(message)s',
                level='DEBUG')
            self.logger = logging.getLogger('DiffCheckerLogIdToPrint')
        elif isinstance(logger, str):
            logging.shutdown()
            reload(logging)
            handler = logging.FileHandler(logger, mode='w+')
            handler.setFormatter(logging.Formatter(
                fmt='%(levelname)s|%(asctime)s|%(message)s'))
            self.logger = logging.getLogger('DiffCheckerLogIdToDump')
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
        else:
            # if external logger provided
            self.logger = logger
        self.log_level = qa_log_level or 30
        self.log_info = log_info

        qa_levels = {
            'loose':{
                'stats':['mean', ch.na_rate],
                'threshold':.5
            },
            'mid':{
                'stats':['mean', 'std', ch.na_rate],
                'threshold':.2
            },
            'strict':{
                'stats':['mean', 'std', 'count', 'min', 'max', ch.na_rate],
                'threshold':.1
            }
        }
        if qa_level not in qa_levels.keys():
            raise ValueError('`qa_level` not right, choose one of {}'\
                .format(qa_levels.keys()))
        self.set_stats(qa_levels[qa_level]['stats'])
        self.set_threshold(qa_levels[qa_level]['threshold'])

    def set_stats(self, funcs):
        '''Sets statistic functions list to check by.

        Args:
            funcs (list): list of functions and/or function names,
                e.g. [np.sum, 'mean']

        See Also:
            `add_stat <#identifiers.DiffChecker.add_stat>`_: just to add one

        '''
        if not self.df_fit_stats.empty:
            raise ValueError('self.stats cannot be altered after `fit()` call')
        if not isinstance(funcs, list):
            raise TypeError('`funcs` must be a list')
        self._method_init_logger(locals())

        self.stats = funcs

    def add_stat(self, func):
        '''Appends a statistic function into the existing list (i.e. `stats <#identifiers.DiffChecker.stats>`_).

        Args:
            func (func): function name (e.g. np.sum or 'mean')

        See Also:
            `set_stats <#identifiers.DiffChecker.set_stats>`_: to reset all

        '''
        if not self.df_fit_stats.empty:
            raise ValueError('self.stats cannot be altered after `fit()` call')
        if not (isinstance(func, str) or callable(func)):
            raise TypeError('`func` must be str or callable')
        if func in self.stats:
            raise ValueError('`func` is already in `self.stats`')
        self._method_init_logger(locals())

        self.stats.append(func)

    def set_threshold(self, threshold):
        '''Sets threshold for statistic-column pairs.

        Args:
            threshold (float or dict): can be used to set for all or column
                statistic pairs.

        Example:
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

        '''
        self._method_init_logger(locals())

        if isinstance(threshold, dict):
            if self.df_fit_stats.empty:
                raise ValueError('call `fit()` first for column level threshold')
            for col, v1 in threshold.items():
                if col not in self.df_fit_stats.columns:
                    raise ValueError('{} not found in fitted DataFrame'\
                        .format(col))
                if isinstance(v1, dict):
                    for stat, v2 in v1.items():
                        if stat not in self.df_fit_stats.index:
                            raise ValueError(
                                "'{0}' not set as stat, available stats are {1}"\
                                .format(stat, self.df_fit_stats.index.tolist()))
                        th = float(v2)
                        assert th >= 0
                        self.threshold_df.loc[stat, col] = th
                else:
                    th = float(v1)
                    assert th >= 0
                    self.threshold_df.loc[:, col] = th
        else:
            th = float(threshold)
            assert th >= 0
            self.threshold = th

    def fit(self, df):
        '''Fits given `df`.

        Based on given `df` and `stats <#identifiers.DiffChecker.stats>`_ attribute, this method constructs
        `df_fit_stats <#identifiers.DiffChecker.df_fit_stats>`_ attribute to store column statistics. This is later to
        be used by `check <#identifiers.DiffChecker.check>`_ method. Only works
        in numerical columns.

        Args:
            df (pd.DataFrame): data to be fit

        Example:
            >>> dc = DiffChecker()
            >>> dc.set_stats(['mean', 'max'])
            >>> dc.fit(pd.DataFrame({'col1':[1, 2, 3, 4], 'col2':[0]*4}))
            >>> print(dc.df_fit_stats)
                  col1  col2
            mean   2.5   0.0
            max    4.0   0.0

        '''
        assert isinstance(self.stats, list) and len(self.stats) >= 1
        if not isinstance(df, pd.DataFrame):
            raise TypeError('`df` must be a pd.DataFrame')
        self._method_init_logger(locals())

        self.df_fit_stats = pd.DataFrame()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                for stat in self.stats:
                    if isinstance(stat, str):
                        stat_name = stat
                    else:
                        stat_name = stat.__name__
                    self.df_fit_stats.loc[stat_name, col] = df[col].agg(stat)

        self.threshold_df = self.df_fit_stats.copy()
        self.threshold_df.replace(self.threshold_df, np.NaN, inplace=True)

    def check(self, df_to_check, columns=None, columns_to_exclude=None):
        '''Checks given `df_to_check` based on fitted `df` stats.

        For each column stat pairs, it checks if stat is in given threshold by
        utilizing `qa_array_statistics <checkers.html#checkers.qa_array_statistics>`_.
        If any stat qa fails, returns `False`, `True otherwise`.

        Args:
            df_to_check (pd.DataFrame): data to check
            columns (None or list): if given, only these columns will be
                considered for qa
            columns_to_exclude (None or list): columns to exclude from qa

        Returns:
            bool: is QA passed or not

        Example:
            >>> dc = DiffChecker()
            >>> dc.set_threshold(0.2)
            >>> dc.set_stats(['mean', 'max', np.sum])
            >>> dc.fit(pd.DataFrame({'col1':[1, 2, 3, 4], 'col2':[1]*4}))
            >>> dc.check(pd.DataFrame({'col1':[1, 2, 3, 4], 'col2':[0]*4}))
            False
            >>> dc.check(pd.DataFrame({'col1':[1, 2.1, 3.2, 4.2], 'col2':[1.1]*4}))
            True

        '''
        assert isinstance(self.stats, list) and len(self.stats) >= 1
        if not isinstance(df_to_check, pd.DataFrame):
            raise TypeError('`df_to_check` must be a pd.DataFrame')
        if columns is not None and columns_to_exclude is not None:
            raise ValueError('only one must be given, '
                             '`columns` or `columns_to_exclude`')
        if columns is not None:
            if not isinstance(columns, list):
                raise TypeError('`columns` must be a list')
        if columns_to_exclude is not None:
            if not isinstance(columns_to_exclude, list):
                raise TypeError('`columns_to_exclude` must be a list')
        self._method_init_logger(locals())

        cols_to_check = self.df_fit_stats.columns.tolist()
        if columns:
            cols_to_check = list(set(cols_to_check) & set(columns))
        if columns_to_exclude:
            cols_to_check = [c for c in cols_to_check if c not \
                in columns_to_exclude]

        qa_results = []
        for col in cols_to_check:
            for stat in self.stats:
                if isinstance(stat, str):
                    stat_name = stat
                else:
                    stat_name = stat.__name__
                th = self.threshold_df.loc[stat_name, col]
                th = self.threshold if pd.isna(th) else th
                val = self.df_fit_stats.loc[stat_name, col]
                tol = abs(val)*th
                ll, ul = val-tol, val+tol
                result = ch.qa_array_statistics(
                    df_to_check[col],
                    {stat:[ll, ul]},
                    logger=self.logger,
                    log_level=self.log_level,
                    name=col)
                qa_results.append(result)

        return all(qa_results)

    def to_pickle(self, path='DiffChecker.pkl'):
        '''Pickle (serialize) object to a file.

        Args:
            path (str): file path where the pickled object will be stored

        Example:
            To save a `*.pkl` file:

            >>> dc1 = DiffChecker()
            >>> dc1.fit(pd.DataFrame({'col1':[1, 2, 3, 4], 'col2':[0]*4}))
            >>> dc1.to_pickle(path='DiffChecker.pkl')

            To load the same object later:

            >>> import pickle
            >>> pkl_file = open('DiffChecker.pkl', 'rb')
            >>> dc2 = pickle.load(pkl_file)
            >>> pkl_file.close()
            >>> os.remove('DiffChecker.pkl')

        '''
        self._method_init_logger(locals())
        self.logger = None
        output = open(path, 'wb')
        pickle.dump(self, output, -1)
        output.close()

    def _method_init_logger(self, args, exclude=['self']):
        '''Logs method initiation with given arguments.

        Args:
            args (dict): local arguments, i.e. `locals()`
            exclude (list): arguments to exclude, e.g. `self`

        '''
        if self.logger and self.log_info:
            method_name = sys._getframe(1).f_code.co_name
            self.logger.info("{} initiated.".format(method_name))
            for k, v in args.items():
                if k not in exclude:
                    self.logger.info(method_name+' locals: '+k+'='+str(v)[:100])

if __name__ == "__main__":
    import doctest
    doctest.testmod()
