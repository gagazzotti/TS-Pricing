"""TBD"""

import unittest
import warnings

import numpy as np
from fypy.model.levy.TemperedStable import TemperedStable
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward

from src.mellin_ts.pricing.TSPricer import TemperedStablePricer


class TestMellinTS(unittest.TestCase):
    """TBD"""

    def test_double_sided(self):
        """TBD"""
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
        strikes = np.arange(1.2, 1.5, 0.1)
        ttms = np.arange(0.7, 1.5, 0.1)
        log10_error = []
        for strike in strikes:
            for ttm in ttms:
                option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
                ts_p_pricer = TemperedStablePricer(**ts_params)
                price = ts_p_pricer.price(**option_params, N=80)
                disc_curve = DiscountCurve_ConstRate(rate=option_params["r"])
                div_disc = DiscountCurve_ConstRate(rate=option_params["q"])
                fwd = EquityForward(S0=1, discount=disc_curve, divDiscount=div_disc)
                bg_model = TemperedStable(fwd, disc_curve, **ts_params)
                proj_pricer = ProjEuropeanPricer(
                    model=bg_model, N=2**17, order=3, alpha_override=100
                )
                proj_price = proj_pricer.price(T=ttm, K=strike, is_call=True)
                log10_error.append(np.log10(np.abs(price - proj_price)))
                self.assertAlmostEqual(price, proj_price)
        print(
            f"\n Mellin One sided => Ok! (Error max: {np.round(np.max(log10_error), 3)})"
        )


if __name__ == "__main__":
    unittest.main()
