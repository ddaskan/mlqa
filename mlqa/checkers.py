'''
This script includes indivdual QA functions for the module.
'''
from itertools import combinations
import pandas as pd

def qa_df_set(
        dfs, threshold=.1, ignore_min=None, ignore_max=None,
        stats_to_exclude=None, columns_to_exclude=None, error_columns=None,
        logger=None, note=None):
    '''
    Wrapper for `qa_df_pair()` to apply 2 length subsequences of `dfs`.

    QA datasets' statistics by utilizing describe() method
    of pd.DataFrame. Ignores non-numeric columns.

    Args:
        dfs: set of pd.DataFrame
        threshold: float, percentage threshold for absolute
            percentage error between statistics
        ignore_min: None or float, ignore stats less or equal than this to
            handle division errors or extreme values
        ignore_max: None or float, ignore stats greater or equal than this to
            handle extreme values
        stats_to_exclude: None or list, statistics to exclude
            as list of strings,
            e.g. ['count', 'mean', 'std', 'min', '25%', '50%',
            '75%', 'max']
        columns_to_exclude: None or list, columns to exclude
            as list of strings
        error_columns: None or list, error columns for error if any
            if given, then test results for non error columns
            would be ignored. Only these columns are logged with
            level 40.
        logger: Python logging object or None
        note: None or str, additional information for `logger`

    Returns:
        bool, is QA passed or not
    '''
    if not all([isinstance(df, pd.DataFrame) for df in dfs]):
        raise TypeError('elements of `dfs` must be pd.DataFrame')

    pairs = combinations(dfs, 2)
    qa_results = []
    for pair in pairs:
        result = qa_df_pair(
            pair[0], pair[1], threshold=threshold, ignore_min=ignore_min,
            ignore_max=ignore_max, stats_to_exclude=stats_to_exclude,
            columns_to_exclude=columns_to_exclude, error_columns=error_columns,
            logger=logger, note=note)
        qa_results.append(result)

    return all(qa_results)

def qa_df_pair(
        df1, df2, threshold=.1, ignore_min=None, ignore_max=None,
        stats_to_exclude=None, columns_to_exclude=None, error_columns=None,
        logger=None, note=None):
    '''
    QA two datasets' statistics by utilizing describe() method
    of pd.DataFrame. Ignores non-numeric columns.

    Args:
        df1: pd.DataFrame, test dataframe
        df2: pd.DataFrame, test dataframe
        threshold: float, percentage threshold for absolute
            percentage error between statistics
        ignore_min: None or float, ignore stats less or equal than this to
            handle division errors or extreme values
        ignore_max: None or float, ignore stats greater or equal than this to
            handle extreme values
        stats_to_exclude: None or list, statistics to exclude
            as list of strings,
            e.g. ['count', 'mean', 'std', 'min', '25%', '50%',
            '75%', 'max']
        columns_to_exclude: None or list, columns to exclude
            as list of strings
        error_columns: None or list, error columns for error if any
            if given, then test results for non error columns
            would be ignored. Only these columns are logged with
            level 40.
        logger: Python logging object or None
        note: None or str, additional information for `logger`

    Returns:
        bool, is QA passed or not
    '''
    if not (isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame)):
        raise TypeError('`df1` and `df2` must be pd.DataFrame')

    details = str(threshold)
    if note:
        details += ' for ' + note
    if logger:
        logger.info('train/test sets QA initiated with threshold ' + details)

    df1_describe = df1.describe()
    df2_describe = df2.describe()

    if columns_to_exclude:
        df1_describe = df1_describe.drop(
            columns_to_exclude,
            axis=1,
            errors='ignore')
        df2_describe = df2_describe.drop(
            columns_to_exclude,
            axis=1,
            errors='ignore')

    if error_columns:
        for e_col in error_columns:
            if e_col not in df1_describe.columns:
                raise KeyError('`{}` not in `{}`'.format(
                        e_col, 
                        df1_describe.columns.tolist()))

    if stats_to_exclude:
        df1_describe = df1_describe.drop(stats_to_exclude, axis=0)
        df2_describe = df2_describe.drop(stats_to_exclude, axis=0)

    results = (df1_describe == df2_describe) | \
        (abs(df1_describe - df2_describe)/df1_describe <= threshold)
    warn_results = []
    error_results = []

    for i, res in results.iterrows():
        for col in results.columns:
            if ignore_min is not None and \
                df1_describe.loc[i, col] <= ignore_min and \
                df2_describe.loc[i, col] <= ignore_min:
                continue
            if ignore_max is not None and \
                df1_describe.loc[i, col] >= ignore_max and \
                df2_describe.loc[i, col] >= ignore_max:
                continue

            msg = i + ' of ' + col + ' not passed. Values are ' \
                + str(round(df1_describe.loc[i, col], 5)) + ' and ' \
                + str(round(df2_describe.loc[i, col], 5))

            if error_columns and col in error_columns:
                error_results.append(res[col])
                if not res[col]:
                    if logger:
                        logger.log(40, msg)

            else:
                warn_results.append(res[col])
                if not res[col]:
                    if logger:
                        logger.log(30, msg)

    if logger:
        logger.info('train/test sets QA done with threshold ' + details)

    if error_columns:
        return all(error_results)
    return all(warn_results)

def qa_preds(preds, warn_range, error_range=None, logger=None, note=None):
    '''
    Wrapper for `qa_array_statistics` for stats `min` and `max` only

    It should be mainly used to also log QA steps and prediction
    statistics. Use `qa_array_statistics` for detailed QA on
    prediction array.

    Args:
        preds: pd.Series
        warn_range: 2 elements iterable, e.g. [min, max] to warn
        error_range: 2 elements iterable or None, e.g. [min, max]
            for error, should involve warn_range.
            If not None, QA result by `warn_range` is ignored.
        logger: Python logging object or None. If None,
            no practical use of this function. Use
            `qa_array_statistics` instead.
        note: str, additional information for `logger`

    Returns:
        bool, is QA passed or not
    '''
    if not isinstance(preds, pd.Series):
        raise TypeError('`preds` must be a pd.Series')
    if not warn_range[1] > warn_range[0]:
        raise ValueError(
            '`warn_range` not right, must be `warn_range[1]` > `warn_range[0]`')
    if error_range:
        if not (error_range[1] > warn_range[1] and \
            error_range[0] < warn_range[0]):
            raise ValueError('`error_range` must contain `warn_range`')

    preds_stats = {k:round(v, 5) for k, v in pd.Series(preds).describe().items()}
    details = str(warn_range)
    if note:
        details += ' for ' + note
    if logger:
        logger.info('predictions QA initiated with warn_range ' + details)
        logger.info('predictions statistics: ' + str(preds_stats))

    is_passed = qa_array_statistics(
        preds,
        stats={
            'min':[warn_range[0], None],
            'max':[None, warn_range[1]]},
        logger=logger,
        log_level=30)

    if error_range:
        is_passed = qa_array_statistics(
            preds,
            stats={
                'min':[error_range[0], None],
                'max':[None, error_range[1]]},
            logger=logger,
            log_level=40)

    if logger:
        logger.info('predictions QA done with warn_range ' + details)

    return is_passed

def qa_category_distribution_on_value(
        df, category_column_name, distribution, value_column_name,
        tolerance=.1, logger=None, log_level=30):
    '''
    QA check for the distribution of category-value pairs in a pd.DataFrame

    Args:
        df: pd.DataFrame
        category_column_name: str, column name for the category e.g. 'Gender'
        distribution: dict, expected value distribution of the category
            (e.g. {'Male':.05, 'Female':.14, 'Undefined':.81})
        value_column_name: str, numeric column name to check distribution
        tolerance: float
        logger: Python logging object or None
        log_level: int,
            https://docs.python.org/3/library/logging.html#logging-levels

    Returns:
        bool, is QA passed or not
    '''
    if not isinstance(df, pd.DataFrame):
        raise TypeError('`df` must be a pd.DataFrame')
    if not isinstance(distribution, dict):
        raise TypeError('`distribution` must be a dict')
    float(tolerance)

    qa_results = []
    df_dist = df[[category_column_name, value_column_name]] \
        .groupby(category_column_name) \
        .sum().reset_index().copy()
    df_dist[value_column_name] = df_dist[value_column_name] \
        /df_dist[value_column_name].sum()

    for cat_value in list(distribution.keys()):
        is_passed = None
        actual = df_dist.loc[df_dist[category_column_name] == \
            cat_value, value_column_name].iloc[0]
        expected = distribution[cat_value]
        log_msg = "{0} distribution looks wrong, check {1} for {0}={2}."\
            " Expected={3}, Actual={4}"\
            .format(
                category_column_name, value_column_name, 
                cat_value, expected, actual)

        if not abs(actual - expected)/expected < tolerance:
            is_passed = False
            if logger:
                logger.log(log_level, log_msg)
        else:
            is_passed = True

        qa_results.append(is_passed)

    return all(qa_results)

def qa_preds_by_metric(
        y_true, y_pred, metric, check_range, logger=None, log_level=30):
    '''
    QA check for model's predictions by selected metric (e.g. R2, AUC)

    Args:
        y_true: array, shape (n_samples, 1)
        y_pred: array, shape (n_samples, 1)
        metric: sklearn like metrics, greater is always better.
            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        check_range: list of 2 float, i.e. [`lower_limit`, `upper_limit`],
            either of elements can be None if no limit is set for that direction.
        logger: Python logging object or None
        log_level: int,
            https://docs.python.org/3/library/logging.html#logging-levels

    Returns:
        bool, is QA passed or not
    '''
    score = metric(y_true, y_pred)
    is_passed = is_value_in_range(
        score, check_range, logger, log_level,
        log_msg='model score (i.e. {}={}) is not in the range of {}' \
            .format(metric.__name__, score, check_range))
    return is_passed

def qa_array_statistics(array, stats, logger=None, log_level=30):
    '''
    QA check for 1D array statistics such as mean, count.

    Args:
        array: array, shape (n_samples, 1)
        stats: dict, stats to qa
            (e.g. {'mean':[0.1, 0.99], 'count':[100, None]}(
            Options for keys are ['mean', 'min', 'max', 'sum', 'count', 'std'].
        logger: Python logging object or None
        log_level: int,
            https://docs.python.org/3/library/logging.html#logging-levels

    Returns:
        bool, is QA passed or not
    '''
    stats_options = ['mean', 'min', 'max', 'sum', 'count', 'std']
    if not all([func in stats_options for func in stats.keys()]):
        raise ValueError('given stat not in {}'.format(stats_options))

    array_copy = pd.Series(array).copy()
    qa_results_for_stats = []

    for func in stats.keys():
        check_range = stats[func]
        value = array_copy.agg(func)
        is_passed = is_value_in_range(
            value, check_range, logger, log_level,
            log_msg='{} value (i.e. {}) is not in the range of {}' \
                .format(func, value, check_range))
        qa_results_for_stats.append(is_passed)

    return all(qa_results_for_stats)

def is_value_in_range(
        value, check_range, logger=None, log_level=None, log_msg=None):
    '''
    Checks if a `value` is in given `check_range`.
    '''
    float(value)
    iter(check_range)
    if check_range[0] and check_range[1]:
        if not check_range[0] <= check_range[1]:
            raise ValueError(
                '{} is wrong must be `check_range[0]` <= `check_range[1]`'\
                .format(check_range))

    is_passed = True

    if not log_msg:
        log_msg = 'value (i.e. {}) is not in the range of {}' \
        .format(value, check_range)

    if check_range[0] is not None:
        if not check_range[0] <= value:
            is_passed = False
            if logger:
                logger.log(log_level, log_msg)

    if check_range[1] is not None:
        if not check_range[1] >= value:
            is_passed = False
            if logger:
                logger.log(log_level, log_msg)

    return is_passed
