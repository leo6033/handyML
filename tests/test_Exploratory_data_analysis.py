import unittest
from sklearn.utils.estimator_checks import check_estimator
import handyML

class TestOverview(unittest.TestCase):
    def test_check_estimator(self):
        assert check_estimator(handyML.Exploratory_data_analysis.Overview) is None

