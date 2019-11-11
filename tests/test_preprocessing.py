import unittest
from sklearn.utils.estimator_checks import check_estimator
import handyML

class TestBetaEncoder(unittest.TestCase):
    def test_check_estimator(self):
        assert check_estimator(handyML.preprocessing.BetaEncoder) is None

