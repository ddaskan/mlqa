import unittest
import sys
import logging
import pandas as pd
import numpy as np
sys.path.append('../')
from mlqa.identifier import DiffChecker

class TestDiffChecker(unittest.TestCase):

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

    @unittest.skip("not ready")
    def test___init__(self):
        pass

    def test_set_stats(self):
        dc = DiffChecker()
        self.assertRaises(TypeError, dc.set_stats)

        dc = DiffChecker()
        dc.df_fit_stats = self.df
        self.assertRaises(ValueError, dc.set_stats, *[['mean']])

        dc = DiffChecker()
        self.assertRaises(TypeError, dc.set_stats, *['mean'])
        self.assertRaises(TypeError, dc.set_stats, *[1])

        with self.assertLogs(self.logger_name, level='INFO') as log:
            dc = DiffChecker(logger=self.logger, log_info=True)
            dc.set_stats(['mean', 'std'])
        self.assertListEqual(
            log.output[-2:], 
            [
                "INFO:test_mlqa:set_stats initiated.",
                "INFO:test_mlqa:set_stats locals: funcs=['mean', 'std']"
            ])

        dc = DiffChecker()
        dc.set_stats(['mean', 'count'])
        self.assertListEqual(dc.stats, ['mean', 'count'])

        dc = DiffChecker()
        dc.set_stats(['mean', np.std])
        self.assertListEqual(dc.stats, ['mean', np.std])

    def test_add_stat(self):
        dc = DiffChecker()
        self.assertRaises(TypeError, dc.add_stat)

        dc = DiffChecker()
        dc.df_fit_stats = self.df
        self.assertRaises(ValueError, dc.add_stat, *['mean'])

        dc = DiffChecker()
        self.assertRaises(TypeError, dc.add_stat, *[['mean']])
        self.assertRaises(TypeError, dc.add_stat, *[1])

        with self.assertLogs(self.logger_name, level='INFO') as log:
            dc = DiffChecker(logger=self.logger, log_info=True)
            dc.add_stat('std')
        self.assertListEqual(
            log.output[-2:], 
            [
                "INFO:test_mlqa:add_stat initiated.",
                "INFO:test_mlqa:add_stat locals: func=std"
            ])

        dc = DiffChecker()
        dc.set_stats(['mean', 'std'])
        dc.add_stat('count')
        self.assertIn('count', dc.stats)

        dc = DiffChecker()
        dc.set_stats(['mean', 'count'])
        dc.add_stat(np.sum)
        self.assertIn(np.sum, dc.stats)

    def test_set_threshold(self):
        dc = DiffChecker()
        self.assertRaises(TypeError, dc.set_threshold)       

        dc = DiffChecker()
        self.assertRaises(AssertionError, dc.set_threshold, *[-.1])

        dc = DiffChecker()
        self.assertRaises(ValueError, dc.set_threshold, *[{}])

        dc = DiffChecker()
        dc.set_stats(['mean', 'count'])
        dc.fit(self.df)
        self.assertRaises(ValueError, dc.set_threshold, *[{'Err_Col':.1}])
        self.assertRaises(
            ValueError, dc.set_threshold, *[{'Fare':{'max':.1}}])
        self.assertRaises(
            AssertionError, dc.set_threshold, *[{'Fare':{'mean':-.1}}])
        self.assertRaises(
            ValueError, dc.set_threshold, *[{'Fare':{'mean':'err'}}])

        dc = DiffChecker()
        dc.fit(self.df)
        self.assertRaises(ValueError, dc.set_threshold, *['err'])
        self.assertRaises(AssertionError, dc.set_threshold, *[-.1])

        with self.assertLogs(self.logger_name, level='INFO') as log:
            dc = DiffChecker(logger=self.logger, log_info=True)
            dc.fit(self.df)
            dc.set_threshold(0.2)
        self.assertListEqual(
            log.output[-2:], 
            [
                "INFO:test_mlqa:set_threshold initiated.",
                "INFO:test_mlqa:set_threshold locals: threshold=0.2"
            ])

        dc = DiffChecker()
        dc.set_stats(['mean', 'count', 'std', 'max'])
        dc.fit(self.df)
        dc.set_threshold(0.3)
        self.assertEqual(0.3, dc.threshold)

        dc.set_threshold(0.0)
        self.assertEqual(0.0, dc.threshold)

        dc.set_threshold({'Fare':0.85})
        self.assertTrue(
            all([th==0.85 for th in dc.threshold_df['Fare'].tolist()]))

        dc.set_threshold({'Survived':{'std':0.95}, 'Pclass':0.35})
        self.assertTrue(pd.isna(dc.threshold_df['Survived'][0]))
        self.assertTrue(pd.isna(dc.threshold_df['Survived'][1]))
        self.assertEqual(0.95, dc.threshold_df['Survived'][2])
        self.assertTrue(pd.isna(dc.threshold_df['Survived'][3]))
        self.assertTrue(
            all([th==0.35 for th in dc.threshold_df['Pclass'].tolist()]))

    @unittest.skip("not ready")
    def test_fit(self):
        pass

    @unittest.skip("not ready")
    def test_check(self):
        pass

    @unittest.skip("not ready")
    def test_to_pcikle(self):
        pass

    def test__method_init_logger(self):
        # no need to write cases for this method since it's already 
        # being tested in other cases
        pass

if __name__ == '__main__':
    unittest.main()
