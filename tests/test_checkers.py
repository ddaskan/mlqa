import unittest
import sys
import logging
import pandas as pd
sys.path.append('../')
from mlqa import checkers

class TestCheckers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv('titanic.csv')
        cls.df1 = cls.df.iloc[:100]
        cls.df2 = cls.df.iloc[100:200]
        cls.df3 = cls.df.iloc[200:300]
        cls.df4 = cls.df.iloc[300:310]

        cls.logger_name = 'test_mlqa'
        logging.basicConfig(format='%(asctime)-15s %(message)s', level='DEBUG')
        cls.logger = logging.getLogger(cls.logger_name)

    def test_qa_df_set(self):
        func = checkers.qa_df_set

        self.assertRaises(TypeError, func, *[[pd.DataFrame(), 'error']])

        self.assertTrue(
            func([self.df1, self.df2], threshold=.35, ignore_min=8.0))
        self.assertTrue(
            func(
                [self.df1, self.df2, self.df3],
                threshold=.35,
                ignore_min=8.0,
                columns_to_exclude=['Fare']))
        self.assertTrue(
            func(
                [self.df1, self.df2],
                threshold=.35,
                stats_to_exclude=['min', 'max', '75%']))
        self.assertTrue(
            func(
                [self.df1, self.df2, self.df3],
                threshold=.001,
                stats_to_exclude=['mean', 'std', 'min', '25%', '50%', '75%', 'max']))
        self.assertTrue(
            func(
                [self.df1, self.df2, self.df3, self.df4],
                threshold=.2,
                ignore_min=550))

        self.assertFalse(
            func([self.df1, self.df2, self.df3], threshold=.35, ignore_min=8.0))
        self.assertFalse(
            func(
                [self.df1, self.df2, self.df4],
                threshold=.35,
                stats_to_exclude=['min', 'max', '75%']))
        self.assertFalse(
            func(
                [self.df1, self.df2, self.df3, self.df4],
                threshold=.001,
                stats_to_exclude=['mean', 'std', 'min', '25%', '50%', '75%', 'max']))
        self.assertFalse(func([self.df1, self.df2, self.df3, self.df4]))
        self.assertFalse(
            func(
                [self.df1, self.df2, self.df3, self.df4],
                threshold=.2,
                ignore_min=50))

    def test_qa_df_pair(self):
        func = checkers.qa_df_pair

        self.assertRaises(TypeError, func)
        self.assertRaises(TypeError, func, *[pd.DataFrame(), 'error'])
        self.assertRaises(TypeError, func, *['error', pd.DataFrame()])
        self.assertRaises(TypeError, func, *['error', 'error'])
        self.assertRaises(
            KeyError, 
            func, 
            **{'df1':self.df1, 'df2':self.df2, 'error_columns':['error_col']})

        with self.assertLogs(self.logger_name, level='INFO') as log:
            func(
                self.df1,
                self.df2,
                threshold=.2,
                error_columns=['Fare', 'Age'],
                logger=self.logger)
        self.assertEqual(
            log.output,
            [
                'INFO:test_mlqa:train/test sets QA initiated with threshold 0.2',
                'WARNING:test_mlqa:mean of Survived not passed. Values are 0.41 and 0.28',
                'WARNING:test_mlqa:std of Parents/Children Aboard not passed.'
                ' Values are 0.96735 and 0.76877',
                'ERROR:test_mlqa:std of Fare not passed. Values are 40.97291 and 31.94107',
                'ERROR:test_mlqa:min of Age not passed. Values are 0.83 and 1.0',
                'ERROR:test_mlqa:min of Fare not passed. Values are 7.225 and 0.0',
                'WARNING:test_mlqa:75% of Parents/Children Aboard not passed.'
                ' Values are 0.0 and 1.0',
                'WARNING:test_mlqa:max of Siblings/Spouses Aboard not passed.'
                ' Values are 5.0 and 8.0',
                'INFO:test_mlqa:train/test sets QA done with threshold 0.2',
            ])

        self.assertTrue(
            func(
                self.df1,
                self.df2,
                threshold=.35,
                stats_to_exclude=['min', 'max', '75%']))
        self.assertTrue(
            func(
                self.df1,
                self.df2,
                threshold=.35,
                columns_to_exclude=[
                    'Parents/Children Aboard', 'Siblings/Spouses Aboard', 'Fare']))
        self.assertTrue(func(self.df1, self.df2, threshold=.35, ignore_min=8.0))
        self.assertTrue(func(self.df1, self.df2, threshold=.2, ignore_max=0.0))
        self.assertTrue(
            func(
                self.df1,
                self.df2,
                threshold=.05,
                stats_to_exclude=['min', 'max', '75%'],
                error_columns=['Pclass']))

        self.assertFalse(func(self.df1, self.df2, threshold=.2))
        self.assertFalse(
            func(
                self.df1,
                self.df2,
                threshold=.1,
                columns_to_exclude=[
                    'Parents/Children Aboard', 'Siblings/Spouses Aboard', 'Fare']))
        self.assertFalse(
            func(
                self.df1,
                self.df2,
                threshold=.05,
                stats_to_exclude=['min', 'max', '75%']))

    def test_qa_preds(self):
        func = checkers.qa_preds

        self.assertRaises(TypeError, func)
        self.assertRaises(TypeError, func, *[list(), [1, 5], [0, 10]])
        self.assertRaises(ValueError, func, *[pd.Series(range(2, 4)), [1, -5], [0, 10]])
        self.assertRaises(ValueError, func, *[pd.Series(range(2, 4)), [1, 5], [2, 10]])
        self.assertRaises(ValueError, func, *[pd.Series(range(2, 4)), [1, 5], [0, 4]])
        self.assertRaises(TypeError, func, *[pd.Series(range(1, 100)), [None, 190]])

        with self.assertLogs(self.logger_name, level='INFO') as log:
            func(pd.Series(range(1, 100)), [10, 90], [5, 95], logger=self.logger)
        self.assertCountEqual(
            log.output[:1]+log.output[2:], # stats dict is unordered so ignored
            [
                'INFO:test_mlqa:predictions QA initiated with warn_range [10, 90]',
                'WARNING:test_mlqa:min value (i.e. 1) is not in the range of [10, None]',
                'WARNING:test_mlqa:max value (i.e. 99) is not in the range of [None, 90]',
                'ERROR:test_mlqa:min value (i.e. 1) is not in the range of [5, None]',
                'ERROR:test_mlqa:max value (i.e. 99) is not in the range of [None, 95]',
                'INFO:test_mlqa:predictions QA done with warn_range [10, 90]'
            ])

        self.assertTrue(func(pd.Series(range(1, 100)), [-1, 190]))
        self.assertTrue(func(pd.Series(range(1, 100)), [10, 90], [0, 100]))

        self.assertFalse(func(pd.Series(range(1, 100)), [10, 190]))
        self.assertFalse(func(pd.Series(range(1, 100)), [10, 90]))
        self.assertFalse(func(pd.Series(range(1, 100)), [10, 90], [0, 98]))

    def test_qa_category_distribution(self):
        func = checkers.qa_category_distribution_on_value

        self.assertRaises(TypeError, func)
        self.assertRaises(
            TypeError,
            func,
            *['error', 'c_col', {'Male':.05}, 'v_col', .1])
        self.assertRaises(
            TypeError,
            func,
            *[pd.DataFrame(), 'c_col', 'error', 'v_col', .1])
        self.assertRaises(
            ValueError,
            func,
            *[pd.DataFrame(), 'c_col', {'Male':.05}, 'v_col', 'error'])
        self.assertRaises(
            IndexError,
            func,
            *[self.df, 'Sex', {'male':.33, 'female_err':.66}, 'Survived']
            )
        self.assertRaises(
            KeyError,
            func,
            *[self.df, 'Sex', {'male':.33, 'female':.66}, 'Survived_err']
            )

        with self.assertLogs(self.logger_name, level='WARN') as log:
            func(self.df, 'Sex', {'male':.03}, 'Survived', logger=self.logger)
            func(self.df, 'Sex', {'male':.1}, 'Fare', logger=self.logger, log_level=40)
        self.assertRegex(
            log.output[0],
            '^(WARNING:test_mlqa:Sex distribution looks wrong, check Survived '
            'for Sex=male. Expected=0.03, Actual=0.31)(.*)')
        self.assertRegex(
            log.output[1],
            '^(ERROR:test_mlqa:Sex distribution looks wrong, check Fare for '
            'Sex=male. Expected=0.1, Actual=0.51)(.*)')

        self.assertTrue(
            func(
                self.df,
                'Sex',
                {'male':.33, 'female':.66},
                'Survived',
                tolerance=.1)
            )
        self.assertFalse(
            func(
                self.df,
                'Sex',
                {'male':.33, 'female':.66},
                'Survived',
                tolerance=.01)
            )

        self.assertTrue(
            func(
                self.df,
                'Sex',
                {'male':.53, 'female':.47},
                'Siblings/Spouses Aboard',
                tolerance=.01)
            )
        self.assertFalse(
            func(
                self.df,
                'Sex',
                {'male':.53, 'female':.47},
                'Siblings/Spouses Aboard',
                tolerance=.001)
            )

        self.assertTrue(
            func(
                self.df,
                'Sex',
                {'male':.5, 'female':.5},
                'Fare',
                tolerance=.1)
            )
        self.assertFalse(
            func(
                self.df,
                'Sex',
                {'male':.5, 'female':.5},
                'Fare',
                tolerance=.001)
            )

    def test_qa_preds_by_metric(self):
        func = checkers.qa_preds_by_metric
        y_true = pd.Series(range(20, 40))
        y_pred = y_true + 1
        metric = lambda x, y: abs(x-y).mean()

        self.assertRaises(TypeError, func)

        with self.assertLogs(self.logger_name, level='WARN') as log:
            func(y_true, y_pred, metric, [0.1, 0.2], logger=self.logger, log_level=30)
            func(y_true, y_pred, metric, [None, 0.2], logger=self.logger, log_level=40)
        self.assertEqual(
            log.output,
            [
                'WARNING:test_mlqa:model score (i.e. <lambda>=1.0) is '
                'not in the range of [0.1, 0.2]',
                'ERROR:test_mlqa:model score (i.e. <lambda>=1.0) is '
                'not in the range of [None, 0.2]'
            ])

        self.assertTrue(func(y_true, y_pred, metric, [1, 2]))
        self.assertTrue(func(y_true, y_pred, metric, [1, 2]))
        self.assertTrue(func(y_true, y_pred, metric, [-10, 20]))

        self.assertFalse(func(y_true, y_pred, metric, [1.01, 2]))
        self.assertFalse(func(y_true, y_pred, metric, [10, 20]))
        self.assertFalse(func(y_true, y_pred, metric, [10, None]))

    def test_qa_array_statistics(self):
        func = checkers.qa_array_statistics

        self.assertRaises(TypeError, func)
        self.assertRaises(
            ValueError,
            func,
            *[range(1, 10), {'mean':[0, None], 'mean2':[0, None]}, None, 30])

        with self.assertLogs(self.logger_name, level='WARN') as log:
            func(pd.Series(range(20, 40)), {'mean':[1, 10]}, logger=self.logger, log_level=30)
            func(pd.Series(range(20, 40)), {'min':[1, 5]}, logger=self.logger, log_level=40)
        self.assertEqual(
            log.output,
            [
                'WARNING:test_mlqa:mean value (i.e. 29.5) is not in the range of [1, 10]',
                'ERROR:test_mlqa:min value (i.e. 20) is not in the range of [1, 5]'
            ])

        self.assertTrue(func(pd.Series(range(20, 40)), {'max':[38, 42]}))
        self.assertTrue(func(pd.Series(range(20, 40)), {'std':[1, None]}))
        self.assertTrue(func(pd.Series(range(20, 21)), {'mean':[20, 20]}))
        self.assertTrue(
            func(
                pd.Series(range(1, 101)),
                {
                    'mean':[49, 51],
                    'count':[None, 110],
                    'min':[0, None],
                    'max':[99, 101]}))

        self.assertFalse(func(pd.Series(range(20, 40)), {'max':[1, 10]}))
        self.assertFalse(func(pd.Series(range(20, 40)), {'count':[None, -10]}))
        self.assertFalse(
            func(
                pd.Series(range(1, 101)),
                {
                    'mean':[49, 51],
                    'count':[None, 110],
                    'min':[5, None],
                    'max':[99, 101]}))
        self.assertFalse(
            func(
                pd.Series(range(1, 101)),
                {
                    'mean':[49, 51],
                    'count':[None, 10],
                    'min':[0, None],
                    'max':[99, 101]}))

    def test_is_value_in_range(self):
        func = checkers.is_value_in_range

        self.assertRaises(TypeError, func)
        self.assertRaises(ValueError, func, *['x', []])
        self.assertRaises(TypeError, func, *[1, None])
        self.assertRaises(ValueError, func, *[1, [2, 1]])

        with self.assertLogs(self.logger_name, level='WARN') as log:
            func(5, [1, 4], logger=self.logger, log_level=30)
            func(5, [1, 4], logger=self.logger, log_level=40)
            func(5, [1, 4], logger=self.logger, log_level=40, log_msg='test')
        self.assertEqual(
            log.output,
            [
                'WARNING:test_mlqa:value (i.e. 5) is not in the range of [1, 4]',
                'ERROR:test_mlqa:value (i.e. 5) is not in the range of [1, 4]',
                'ERROR:test_mlqa:test'
            ])

        self.assertTrue(func(5, [1, 10]))
        self.assertTrue(func(5, [1, None]))
        self.assertTrue(func(5, [None, 10]))
        self.assertTrue(func(1, [1, None]))
        self.assertTrue(func(10, [None, 10]))
        self.assertTrue(func(4.1, [1.2, 4.1]))

        self.assertFalse(func(0.0, [1.2, 4.1]))
        self.assertFalse(func(5.3, [1.2, 4.1]))
        self.assertFalse(func(5.3, [None, 4.1]))
        self.assertFalse(func(-5.3, [1.2, None]))

if __name__ == '__main__':
    unittest.main()
