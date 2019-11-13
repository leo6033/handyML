import unittest
import pandas as pd
import numpy as np 
from sklearn.utils.estimator_checks import check_estimator
import handyML

class TestBetaEncoder(unittest.TestCase):
    def test_with_dates(self):

        X = pd.DataFrame({'A':[1, 1, 2, 4], 'B':[2, 1, 3, 4], 'label':[0, 0, 1, 1]})
        be = handyML.preprocessing.BetaEncoder('A')
        be.fit(X, 'label')
        ans = be.transform(X, 'mean')
        
        np.testing.assert_array_equal(ans, [0, 0, 1, 1])
