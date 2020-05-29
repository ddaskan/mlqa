import unittest
import sys
import os
import shutil
import pickle
import logging
import pandas as pd
import numpy as np
sys.path.append('../')
from mlqa.identifier import DiffChecker
from mlqa import checkers as ch

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

        cls.temp_dir = 'temp/'
        os.mkdir(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test___init__(self):
        dc = DiffChecker()
        self.assertIsInstance(dc.log_level, int)
        self.assertIs(dc.log_info, False)
        self.assertTrue(len(dc.stats) >= 1)
        self.assertIsInstance(dc.threshold, float)
        self.assertTrue(dc.threshold_df.empty)
        self.assertTrue(dc.df_fit_stats.empty)

        dc = DiffChecker(qa_log_level=40, log_info=True)
        self.assertIs(dc.log_level, 40)
        self.assertIs(dc.log_info, True)

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

        dc = DiffChecker()
        dc.set_stats(['mean', np.sum])
        self.assertRaises(ValueError, dc.add_stat, *['mean'])
        self.assertRaises(ValueError, dc.add_stat, *[np.sum])

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

    def test_fit(self):
        dc = DiffChecker()
        self.assertRaises(TypeError, dc.fit)
        self.assertRaises(TypeError, dc.fit, *['error'])

        stats = ['mean', 'count']
        dc.set_stats(stats)
        dc.fit(self.df)
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        # check the shape is expected and same for both
        self.assertCountEqual(num_cols, dc.df_fit_stats.columns)
        self.assertCountEqual(stats, dc.df_fit_stats.index)
        self.assertEqual(len(stats), dc.df_fit_stats.shape[0])
        self.assertEqual(dc.threshold_df.shape, dc.df_fit_stats.shape)
        self.assertTrue(dc.threshold_df.columns.equals(dc.df_fit_stats.columns))
        self.assertTrue(dc.threshold_df.index.equals(dc.df_fit_stats.index))
        
        # check values are fine
        for c in dc.threshold_df.columns:
            for v in dc.threshold_df[c]:
                with self.subTest(v=v):
                    self.assertTrue(pd.isna(v))
        self.assertCountEqual(dc.df_fit_stats.T['count'].unique(), [887])
        self.assertAlmostEqual(
            dc.df_fit_stats.loc['mean', 'Survived'],
            0.3856,
            places=4)
        self.assertAlmostEqual(
            dc.df_fit_stats.loc['mean', 'Pclass'],
            2.3055,
            places=4)
        self.assertAlmostEqual(
            dc.df_fit_stats.loc['mean', 'Parents/Children Aboard'],
            0.3833,
            places=4)
        self.assertAlmostEqual(
            dc.df_fit_stats.loc['mean', 'Fare'],
            32.3054,
            places=4)

    def test_check(self):
        dc = DiffChecker()
        dc.fit(self.df)
        self.assertRaises(TypeError, dc.check)
        self.assertRaises(TypeError, dc.check, *['error'])
        self.assertRaises(ValueError, dc.check, *[self.df1, ['error1'], ['error2']])
        self.assertRaises(
            TypeError, 
            dc.check, 
            **{'df_to_check':self.df1, 'columns':'error'})
        self.assertRaises(
            TypeError, 
            dc.check, 
            **{'df_to_check':self.df1, 'columns_to_exclude':'error'})

        with self.assertLogs(self.logger_name, level='INFO') as log:
            dc = DiffChecker(logger=self.logger, log_info=False)
            dc.set_stats(['mean', ch.na_rate, 'std', 'max'])
            dc.set_threshold(0.3)
            dc.fit(self.df)
            dc.check(self.df1)
        self.assertRegex(
            log.output[0], 
            "^WARNING:test_mlqa:mean value \(i.e. 0.73\) is not in the (.*) "
            "for Siblings/Spouses Aboard$")
        self.assertEqual(
            log.output[1], 
            "WARNING:test_mlqa:max value (i.e. 5) is not in the range of "
            "[5.6, 10.4] for Siblings/Spouses Aboard")
        self.assertEqual(
            log.output[2], 
            "WARNING:test_mlqa:max value (i.e. 263.0) is not in the range "
            "of [358.63044, 666.02796] for Fare")

        dc = DiffChecker()
        dc.set_stats(['mean', ch.na_rate, 'std'])
        dc.set_threshold(0.5)
        dc.fit(self.df)
        self.assertTrue(dc.check(self.df1))
        self.assertTrue(dc.check(self.df2))
        self.assertTrue(dc.check(self.df3))
        self.assertFalse(dc.check(self.df4))

        dc = DiffChecker()
        dc.set_stats(['mean', ch.na_rate, 'std'])
        dc.set_threshold(0.3)
        dc.fit(self.df)
        self.assertTrue(
            dc.check(self.df1, columns_to_exclude=['Siblings/Spouses Aboard']))
        self.assertTrue(dc.check(self.df2, columns=['Survived', 'Pclass']))
        self.assertTrue(dc.check(self.df3))
        self.assertFalse(dc.check(self.df4))

        dc = DiffChecker()
        dc.set_stats(['mean', ch.na_rate, 'std'])
        dc.set_threshold(0.3)
        dc.fit(self.df)
        dc.set_threshold({'Siblings/Spouses Aboard':0.5})
        dc.set_threshold({'Fare':{'na_rate':0.0}})
        self.assertTrue(dc.check(self.df1))
        dc.set_threshold({'Survived':{'mean':0.0}})
        self.assertFalse(dc.check(self.df1))
        dc.set_threshold({'Survived':{'mean':0.01}})
        self.assertFalse(dc.check(self.df1))
        self.assertTrue(dc.check(self.df1, columns_to_exclude=['Survived']))
        self.assertTrue(dc.check(self.df1, columns=['Fare']))

        dc = DiffChecker()
        dc.set_stats(['mean', 'max'])
        dc.set_threshold(0.01)
        dc.fit(self.df)
        self.assertFalse(dc.check(self.df1))
        self.assertFalse(dc.check(self.df2))
        self.assertFalse(dc.check(self.df3))
        self.assertFalse(dc.check(self.df4))

        dc = DiffChecker()
        dc.set_stats(['count'])
        dc.fit(self.df1)
        dc.set_threshold(0.01)
        self.assertTrue(dc.check(self.df2))
        dc.set_threshold(0.5)
        self.assertFalse(dc.check(self.df4))

    def test_to_pickle(self):
        log_file = os.path.join(self.temp_dir, 'temp.log')
        dc1 = DiffChecker(logger=log_file, log_info=True)
        dc1.set_stats(['mean', 'max'])
        dc1.fit(self.df)
        fname = os.path.join(self.temp_dir, 'DiffChecker.pkl')
        dc1.to_pickle(path=fname)

        pkl_file = open(fname, 'rb')
        dc2 = pickle.load(pkl_file)
        pkl_file.close()

        self.assertEqual(dc1.threshold, dc2.threshold)
        self.assertTrue(dc1.threshold_df.equals(dc2.threshold_df))
        self.assertEqual(dc1.stats, dc2.stats)
        self.assertTrue(dc1.df_fit_stats.equals(dc2.df_fit_stats))

        dc2.set_threshold({'Fare':0.85})
        self.assertTrue(
            all([th==0.85 for th in dc2.threshold_df['Fare'].tolist()]))
        self.assertRaises(ValueError, dc2.add_stat, *['min'])

    def test__method_init_logger(self):
        # no need to write cases for this method since it's already 
        # being tested in other cases
        pass

if __name__ == '__main__':
    unittest.main()
