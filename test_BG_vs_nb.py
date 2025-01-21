from time import time

import numpy as np
from fypy.model.levy.BilateralGamma import BilateralGamma
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward

from src.mellin_ts.pricing.BGPricer import BGPricer


def main():
    print("##")
    S0, K, r, q, ttm = 1, 1.7, 0, 0, 1
    alphap = 0.58
    lambdap = 1.38
    alpham = 1 / 3
    lambdam = 0.4
    bg_pricer = BGPricer(alphap, lambdap, alpham, lambdam)
    t0 = time()
    price = bg_pricer.price_eur(S0, K, r, q, ttm, N=100)
    price_series = bg_pricer.price_eur_diff(S0, K, r, q, ttm, N=100)
    print("Prices", price, price_series)
    t_series = time() - t0
    print("Price", price_series)
    print("Time", t_series)

    ################
    ################
    print("\n\n#### PROJ ####")
    disc_curve = DiscountCurve_ConstRate(rate=r)
    div_disc = DiscountCurve_ConstRate(rate=q)
    fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)
    bg_model = BilateralGamma(fwd, disc_curve, alphap, lambdap, alpham, lambdam)
    proj_pricer = ProjEuropeanPricer(
        model=bg_model, N=2**17, order=3, alpha_override=100
    )
    times = []
    for _ in range(1):
        t0_proj = time()
        proj_price = proj_pricer.price(T=ttm, K=K, is_call=True)
        t = time() - t0_proj
        times.append(t)
    print(proj_price)
    print("Time proj", np.mean(times), 1.96 * np.std(times) / 1**0.5)
    print("Error", price_series - proj_price)


if __name__ == "__main__":
    main()
