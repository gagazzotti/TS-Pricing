import unittest
import numpy as np
from mellin_ts.OneSidedTSPricer import OneSidedTemperedStablePricer
import time
import warnings


class TestMellinTSp(unittest.TestCase):

    def test_one_sided(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)

        ts_p_params = dict(alpha_p=0.44, beta_p=0.1 + np.exp(1) / 10, lambda_p=1.4)
        strike = 1.5
        ttm = 1.2
        option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
        ts_p_pricer = OneSidedTemperedStablePricer(**ts_p_params)
        price = ts_p_pricer.price(**option_params)
        option_params = dict(S0=1, K=[1.5, 1.6], r=0.02, q=0.05, ttm=[1.2, 1.3])

        price_vect = ts_p_pricer.price_vect(**option_params)

        price_ref = 0.18769860488552348
        self.assertAlmostEqual(price, price_ref)
        # self.assertAlmostEqual(price_vect[0, 0, 0], price_ref)

        print("\n Mellin One sided => Ok!")


if __name__ == "__main__":
    unittest.main()
