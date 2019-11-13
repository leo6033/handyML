import unittest
from sklearn.utils.estimator_checks import check_estimator
import handyML
import pandas as pd
import numpy as np

class TestOverview(unittest.TestCase):
    def test_with_dates(self):
    
        X = pd.DataFrame({'A':[1, 1, np.nan, 4], 'B':[2, 1, 3, 4], 'c':[0, 0, 1, 1]})
        be = handyML.Exploratory_data_analysis.Overview()
        ans = be.check_missing_data(X)
        
        pd.testing.assert_frame_equal(ans, pd.DataFrame({'Total': {'A': 1, 'B': 0, 'c': 0},
                                                         'Percent': {'A': 25.0, 'B': 0.0, 'c': 0.0},
                                                         'Types': {'A': 'float64', 'B': 'int64', 'c': 'int64'}}))