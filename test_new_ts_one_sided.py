"""TBD"""

import unittest
import warnings

import numpy as np
from fypy.model.levy.TemperedStable import TemperedStable
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward

from src.mellin_ts.pricers.onesidedtsnegative_pricer import (
    OneSidedTemperedStablePricerNegative,
)

ts_params = dict(
    alpha_p=0.0,
    beta_p=0.1 + np.exp(1) / 10,
    lambda_p=1.4,
    alpha_m=0.35,
    beta_m=0.5 - np.pi / 100,
    lambda_m=0.4,
)
# strike = 1.3
ttm = 1
strikes = np.arange(0.5, 0.6, 0.1)
ttms = np.arange(1.2, 2, 1)
log10_error = []
for strike in strikes:
    for ttm in ttms:
        print("-----", strike, ttm)
        option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
        # option_params = dict(S0=1, K=0.5, r=0.02, q=0.05, ttm=ttm)
        ts_p_pricer = OneSidedTemperedStablePricerNegative(
            ts_params["alpha_m"], ts_params["beta_m"], ts_params["lambda_m"]
        )
        price = ts_p_pricer.price(**option_params, N=250)
        disc_curve = DiscountCurve_ConstRate(rate=option_params["r"])
        div_disc = DiscountCurve_ConstRate(rate=option_params["q"])
        fwd = EquityForward(S0=1, discount=disc_curve, divDiscount=div_disc)
        bg_model = TemperedStable(fwd, disc_curve, **ts_params)
        proj_pricer = ProjEuropeanPricer(
            model=bg_model, N=2**20, order=3, alpha_override=100
        )
        proj_price = proj_pricer.price(T=ttm, K=strike, is_call=True)
        log10_error.append(np.log10(np.abs(price - proj_price)))
        print("PROJ", proj_price)
        print("SERIE", price)

        print(abs(price - proj_price))


# import numpy as np
# from scipy.special import gamma


# def chf_ts(u, alpha, beta, lam):
#     """
#     Fonction caractéristique du tempered stable.

#     Parameters
#     ----------
#     u : array_like
#     alpha, beta, lam : floats

#     Returns
#     -------
#     complex array
#     """
#     u = np.asarray(u, dtype=np.complex128)
#     return np.exp(alpha * gamma(-beta) * ((lam + 1j * u) ** beta - lam**beta))


# from scipy.integrate import quad


# def cdf_from_chf(x, alpha, beta, lam, u_max=1000, eps=1e-6):
#     """
#     CDF via inversion de la fonction caractéristique.

#     Parameters
#     ----------
#     x : float
#     alpha, beta, lam : floats
#     u_max : cutoff intégration
#     eps : tolérance

#     Returns
#     -------
#     float
#     """

#     def integrand(u):
#         phi = chf_ts(u, alpha, beta, lam)
#         return np.imag(np.exp(-1j * u * x) * phi / u)

#     integral, _ = quad(integrand, eps, u_max, limit=200)

#     return 0.5 - (1 / np.pi) * integral


# print(
#     cdf_from_chf(-0.5, ts_params["alpha_m"], ts_params["beta_m"], ts_params["lambda_m"])
# )
