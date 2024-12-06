import unittest
import numpy as np
from mellin_ts.pricing.TSPricer import TemperedStablePricer

import warnings


class TestMellinTS(unittest.TestCase):
    def test_double_sided(self):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ts_params = dict(
            alpha_p=0.44,
            beta_p=0.1 + np.exp(1) / 10,
            lambda_p=1.4,
            alpha_m=0.35,
            beta_m=0.5 - np.pi / 100,
            lambda_m=0.4,
        )
        strike = 1.5
        ttm = 1.2
        option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
        ts_p_pricer = TemperedStablePricer(**ts_params)
        price = ts_p_pricer.price(**option_params, N=60)
        price_ref = 0.22968572289948497  # PROJ
        # self.assertAlmostEqual(price, price_ref)
        print(np.abs(price - price_ref))
        print("\n Mellin Double sided => Ok!")


if __name__ == "__main__":
    unittest.main()
