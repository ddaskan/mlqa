import unittest
import sys
import os
import shutil
import pickle
import logging
import pandas as pd
import numpy as np
sys.path.append('../')
from mlqa.identifiers import DiffChecker
from mlqa import checkers as ch

class TestDiffChecker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for path in ['', '../', 'tests/', '../tests/']:
            try:
                cls.df = pd.read_csv(path+'titanic.csv')
                break
            except:
                pass
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
        dcr = DiffChecker()
        self.assertIsInstance(dcr.log_level, int)
        self.assertIs(dcr.log_info, False)
        self.assertTrue(len(dcr.stats) >= 1)
        self.assertIsInstance(dcr.threshold, float)
        self.assertTrue(dcr.threshold_df.empty)
        self.assertTrue(dcr.df_fit_stats.empty)

        dcr = DiffChecker(qa_log_level=40, log_info=True)
        self.assertIs(dcr.log_level, 40)
        self.assertIs(dcr.log_info, True)

    def test_set_stats(self):
        dcr = DiffChecker()
        self.assertRaises(TypeError, dcr.set_stats)

        dcr = DiffChecker()
        dcr.df_fit_stats = self.df
        self.assertRaises(ValueError, dcr.set_stats, *[['mean']])

        dcr = DiffChecker()
        self.assertRaises(TypeError, dcr.set_stats, *['mean'])
        self.assertRaises(TypeError, dcr.set_stats, *[1])

        with self.assertLogs(self.logger_name, level='INFO') as log:
            dcr = DiffChecker(logger=self.logger, log_info=True)
            dcr.set_stats(['mean', 'std'])
        self.assertListEqual(
            log.output[-2:],
            [
                "INFO:test_mlqa:set_stats initiated.",
                "INFO:test_mlqa:set_stats locals: funcs=['mean', 'std']"
            ])

        dcr = DiffChecker()
        dcr.set_stats(['mean', 'count'])
        self.assertListEqual(dcr.stats, ['mean', 'count'])

        dcr = DiffChecker()
        dcr.set_stats(['mean', np.std])
        self.assertListEqual(dcr.stats, ['mean', np.std])

    def test_add_stat(self):
        dcr = DiffChecker()
        self.assertRaises(TypeError, dcr.add_stat)

        dcr = DiffChecker()
        dcr.df_fit_stats = self.df
        self.assertRaises(ValueError, dcr.add_stat, *['mean'])

        dcr = DiffChecker()
        self.assertRaises(TypeError, dcr.add_stat, *[['mean']])
        self.assertRaises(TypeError, dcr.add_stat, *[1])

        dcr = DiffChecker()
        dcr.set_stats(['mean', np.sum])
        self.assertRaises(ValueError, dcr.add_stat, *['mean'])
        self.assertRaises(ValueError, dcr.add_stat, *[np.sum])

        with self.assertLogs(self.logger_name, level='INFO') as log:
            dcr = DiffChecker(logger=self.logger, log_info=True)
            dcr.add_stat('std')
        self.assertListEqual(
            log.output[-2:],
            [
                "INFO:test_mlqa:add_stat initiated.",
                "INFO:test_mlqa:add_stat locals: func=std"
            ])

        dcr = DiffChecker()
        dcr.set_stats(['mean', 'std'])
        dcr.add_stat('count')
        self.assertIn('count', dcr.stats)

        dcr = DiffChecker()
        dcr.set_stats(['mean', 'count'])
        dcr.add_stat(np.sum)
        self.assertIn(np.sum, dcr.stats)

    def test_set_threshold(self):
        dcr = DiffChecker()
        self.assertRaises(TypeError, dcr.set_threshold)

        dcr = DiffChecker()
        self.assertRaises(AssertionError, dcr.set_threshold, *[-.1])

        dcr = DiffChecker()
        self.assertRaises(ValueError, dcr.set_threshold, *[{}])

        dcr = DiffChecker()
        dcr.set_stats(['mean', 'count'])
        dcr.fit(self.df)
        self.assertRaises(ValueError, dcr.set_threshold, *[{'Err_Col':.1}])
        self.assertRaises(
            ValueError, dcr.set_threshold, *[{'Fare':{'max':.1}}])
        self.assertRaises(
            AssertionError, dcr.set_threshold, *[{'Fare':{'mean':-.1}}])
        self.assertRaises(
            ValueError, dcr.set_threshold, *[{'Fare':{'mean':'err'}}])

        dcr = DiffChecker()
        dcr.fit(self.df)
        self.assertRaises(ValueError, dcr.set_threshold, *['err'])
        self.assertRaises(AssertionError, dcr.set_threshold, *[-.1])

        with self.assertLogs(self.logger_name, level='INFO') as log:
            dcr = DiffChecker(logger=self.logger, log_info=True)
            dcr.fit(self.df)
            dcr.set_threshold(0.2)
        self.assertListEqual(
            log.output[-2:],
            [
                "INFO:test_mlqa:set_threshold initiated.",
                "INFO:test_mlqa:set_threshold locals: threshold=0.2"
            ])

        dcr = DiffChecker()
        dcr.set_stats(['mean', 'count', 'std', 'max'])
        dcr.fit(self.df)
        dcr.set_threshold(0.3)
        self.assertEqual(0.3, dcr.threshold)

        dcr.set_threshold(0.0)
        self.assertEqual(0.0, dcr.threshold)

        dcr.set_threshold({'Fare':0.85})
        self.assertTrue(
            all([th == 0.85 for th in dcr.threshold_df['Fare'].tolist()]))

        dcr.set_threshold({'Survived':{'std':0.95}, 'Pclass':0.35})
        self.assertTrue(pd.isna(dcr.threshold_df['Survived'][0]))
        self.assertTrue(pd.isna(dcr.threshold_df['Survived'][1]))
        self.assertEqual(0.95, dcr.threshold_df['Survived'][2])
        self.assertTrue(pd.isna(dcr.threshold_df['Survived'][3]))
        self.assertTrue(
            all([th == 0.35 for th in dcr.threshold_df['Pclass'].tolist()]))

    def test_fit(self):
        dcr = DiffChecker()
        self.assertRaises(TypeError, dcr.fit)
        self.assertRaises(TypeError, dcr.fit, *['error'])

        stats = ['mean', 'count']
        dcr.set_stats(stats)
        dcr.fit(self.df)
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        # check the shape is expected and same for both
        self.assertCountEqual(num_cols, dcr.df_fit_stats.columns)
        self.assertCountEqual(stats, dcr.df_fit_stats.index)
        self.assertEqual(len(stats), dcr.df_fit_stats.shape[0])
        self.assertEqual(dcr.threshold_df.shape, dcr.df_fit_stats.shape)
        self.assertTrue(dcr.threshold_df.columns.equals(dcr.df_fit_stats.columns))
        self.assertTrue(dcr.threshold_df.index.equals(dcr.df_fit_stats.index))

        # check values are fine
        for c in dcr.threshold_df.columns:
            for v in dcr.threshold_df[c]:
                with self.subTest(v=v):
                    self.assertTrue(pd.isna(v))
        self.assertCountEqual(dcr.df_fit_stats.T['count'].unique(), [887])
        self.assertAlmostEqual(
            dcr.df_fit_stats.loc['mean', 'Survived'],
            0.3856,
            places=4)
        self.assertAlmostEqual(
            dcr.df_fit_stats.loc['mean', 'Pclass'],
            2.3055,
            places=4)
        self.assertAlmostEqual(
            dcr.df_fit_stats.loc['mean', 'Parents/Children Aboard'],
            0.3833,
            places=4)
        self.assertAlmostEqual(
            dcr.df_fit_stats.loc['mean', 'Fare'],
            32.3054,
            places=4)

    def test_check(self):
        dcr = DiffChecker()
        dcr.fit(self.df)
        self.assertRaises(TypeError, dcr.check)
        self.assertRaises(TypeError, dcr.check, *['error'])
        self.assertRaises(ValueError, dcr.check, *[self.df1, ['error1'], ['error2']])
        self.assertRaises(
            TypeError,
            dcr.check,
            **{'df_to_check':self.df1, 'columns':'error'})
        self.assertRaises(
            TypeError,
            dcr.check,
            **{'df_to_check':self.df1, 'columns_to_exclude':'error'})

        with self.assertLogs(self.logger_name, level='INFO') as log:
            dcr = DiffChecker(logger=self.logger, log_info=False)
            dcr.set_stats(['mean', ch.na_rate, 'std', 'max'])
            dcr.set_threshold(0.3)
            dcr.fit(self.df)
            dcr.check(self.df1)
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

        dcr = DiffChecker()
        dcr.set_stats(['mean', ch.na_rate, 'std'])
        dcr.set_threshold(0.5)
        dcr.fit(self.df)
        self.assertTrue(dcr.check(self.df1))
        self.assertTrue(dcr.check(self.df2))
        self.assertTrue(dcr.check(self.df3))
        self.assertFalse(dcr.check(self.df4))

        dcr = DiffChecker()
        dcr.set_stats(['mean', ch.na_rate, 'std'])
        dcr.set_threshold(0.3)
        dcr.fit(self.df)
        self.assertTrue(
            dcr.check(self.df1, columns_to_exclude=['Siblings/Spouses Aboard']))
        self.assertTrue(dcr.check(self.df2, columns=['Survived', 'Pclass']))
        self.assertTrue(dcr.check(self.df3))
        self.assertFalse(dcr.check(self.df4))

        dcr = DiffChecker()
        dcr.set_stats(['mean', ch.na_rate, 'std'])
        dcr.set_threshold(0.3)
        dcr.fit(self.df)
        dcr.set_threshold({'Siblings/Spouses Aboard':0.5})
        dcr.set_threshold({'Fare':{'na_rate':0.0}})
        self.assertTrue(dcr.check(self.df1))
        dcr.set_threshold({'Survived':{'mean':0.0}})
        self.assertFalse(dcr.check(self.df1))
        dcr.set_threshold({'Survived':{'mean':0.01}})
        self.assertFalse(dcr.check(self.df1))
        self.assertTrue(dcr.check(self.df1, columns_to_exclude=['Survived']))
        self.assertTrue(dcr.check(self.df1, columns=['Fare']))

        dcr = DiffChecker()
        dcr.set_stats(['mean', 'max'])
        dcr.set_threshold(0.01)
        dcr.fit(self.df)
        self.assertFalse(dcr.check(self.df1))
        self.assertFalse(dcr.check(self.df2))
        self.assertFalse(dcr.check(self.df3))
        self.assertFalse(dcr.check(self.df4))

        dcr = DiffChecker()
        dcr.set_stats(['count'])
        dcr.fit(self.df1)
        dcr.set_threshold(0.01)
        self.assertTrue(dcr.check(self.df2))
        dcr.set_threshold(0.5)
        self.assertFalse(dcr.check(self.df4))

    def test_to_pickle(self):
        log_file = os.path.join(self.temp_dir, 'temp.log')
        dcr1 = DiffChecker(logger=log_file, log_info=True)
        dcr1.set_stats(['mean', 'max'])
        dcr1.fit(self.df)
        fname = os.path.join(self.temp_dir, 'DiffChecker.pkl')
        dcr1.to_pickle(path=fname)

        pkl_file = open(fname, 'rb')
        dcr2 = pickle.load(pkl_file)
        pkl_file.close()

        self.assertEqual(dcr1.threshold, dcr2.threshold)
        self.assertTrue(dcr1.threshold_df.equals(dcr2.threshold_df))
        self.assertEqual(dcr1.stats, dcr2.stats)
        self.assertTrue(dcr1.df_fit_stats.equals(dcr2.df_fit_stats))

        dcr2.set_threshold({'Fare':0.85})
        self.assertTrue(
            all([th == 0.85 for th in dcr2.threshold_df['Fare'].tolist()]))
        self.assertRaises(ValueError, dcr2.add_stat, *['min'])

    def test__method_init_logger(self):
        # no need to write cases for this method since it's already
        # being tested in other cases
        pass

if __name__ == '__main__':
    unittest.main()
