import unittest
import numpy as np
from mellin_ts.OneSidedTSPricer import OneSidedTemperedStablePricer


class TestMellinTSp(unittest.TestCase):

    def test_one_sided(self):
        ts_p_params = dict(alpha_p=0.44, beta_p=0.1 + np.exp(1) / 10, lambda_p=1.4)
        strike = 1.5
        ttm = 1.2
        option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
        ts_p_pricer = OneSidedTemperedStablePricer(**ts_p_params)
        price = ts_p_pricer.price(**option_params)
        price_ref = 0.18769860488552348
        self.assertAlmostEqual(price, price_ref)
        print("\n Mellin One sided => Ok!")


if __name__ == "__main__":
    unittest.main()
