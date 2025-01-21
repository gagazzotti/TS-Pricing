"""TBD"""

import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import factorial, gamma, hyp2f1

# pylint: disable=all
from src.gamma_func_cpp.lower_gamma.gamma_incomp import (
    gamma_lower_incomplete_non_normalized as gamma_lower,
)

# pylint: enable=all


warnings.filterwarnings("ignore")


class BGPricer:
    """
    BG pricer
    """

    def __init__(
        self,
        alpha_p: float,
        lambda_p: float,
        alpha_m: float,
        lambda_m: float,
    ):
        self.alpha_p = alpha_p
        self.alpha_m = alpha_m
        self.lambda_p = lambda_p
        self.lambda_m = lambda_m

        self.lambda_ubar = lambda_p + lambda_m
        self.alpha_ubar = 0.5 * (alpha_p + alpha_m)
        self.alpha_bar = 0.5 * (alpha_p - alpha_m)

        self.mbg = self.get_mbg()
        self.zeta = self.zeta_()

        return

    def a(self, alpha: float, beta: float) -> float:
        """a_pm constant in the paper

        Args:
            alpha (float): alpha
            beta (float): beta

        Returns:
            float: apm
        """
        return -alpha * gamma(-beta)

    def zeta_(self):
        """convexity adjustment

        Returns:
            zeta: convex. adj
        """
        zeta_p = ((self.lambda_p / (self.lambda_p - 1)) ** self.alpha_p) * (
            (self.lambda_m / (self.lambda_m + 1)) ** self.alpha_m
        )
        return -np.log(zeta_p)

    def get_mbg(self):
        # Constante densité de BG
        numerator = (self.lambda_p**self.alpha_p) * (self.lambda_m**self.alpha_m)
        denominator = (
            self.lambda_ubar ** (self.alpha_ubar)
            * gamma(self.alpha_p)
            * gamma(self.alpha_m)
            * gamma(1 - self.alpha_p)
        )
        Mbg = numerator / denominator
        return Mbg

    def get_const(self):
        pre = gamma(self.alpha_p + self.alpha_m) / (
            gamma(self.alpha_p)
            * gamma(self.alpha_m + 1)
            * self.lambda_p ** (self.alpha_p + self.alpha_m)
        )
        return pre

    #######################
    #######################
    ## Cash or nothing ####
    #######################
    #######################

    def price_cn(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        ttm: float,
        N: int = 25,
        # time_verbose=True,
    ):
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        serie = self.serie_CN(k, ttm, N)
        return np.exp(-r * ttm) * serie

    ###########################
    ###########################
    ## Eur by diff (AN/CN) ####
    ###########################
    ###########################

    def price_eur_diff(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        ttm: float,
        N: int = 25,
        # time_verbose=True,
    ):
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        CN_value = self.serie_CN(k, ttm, N)
        AN_value = np.exp(k) * self.serie_AN(k, ttm, N)

        return K * np.exp(-r * ttm) * (AN_value - CN_value)

    def serie_CN(
        self,
        k: float | npt.NDArray[np.float64],
        ttm: float | npt.NDArray[np.float64],
        N: int,
    ):
        n_vec = np.arange(0, N)
        fact_n = factorial(n_vec)
        taylor_term = (-1) ** n_vec / fact_n
        # doublon
        am_plus_m = gamma(self.alpha_m + n_vec)
        one_minus_alphap_plus_n = gamma(1 - self.alpha_p + n_vec)

        serie1_1 = 1

        num = (self.lambda_p**self.alpha_p * self.lambda_m**self.alpha_m) * gamma(
            2 * self.alpha_ubar
        )
        denum = -(
            self.lambda_ubar ** (2 * self.alpha_ubar)
            * gamma(self.alpha_p)
            * gamma(self.alpha_m)
            * (self.alpha_p)
        )
        serie1_2 = (num / denum) * hyp2f1(
            2 * self.alpha_ubar,
            1,
            1 + self.alpha_p,
            self.lambda_p / self.lambda_ubar,
        )

        gamma_inc_vec = np.array(
            gamma_lower(
                n_vec + 2 * self.alpha_ubar,
                -k * self.lambda_p * np.ones_like(n_vec),
            )
        )

        serie2 = (
            taylor_term
            * gamma(1 - n_vec - 2 * self.alpha_ubar)
            * am_plus_m
            * self.lambda_ubar ** (self.alpha_ubar + n_vec)
            * self.lambda_p ** (-n_vec - 2 * self.alpha_ubar)
            * gamma_inc_vec
        ).sum() * self.mbg

        gamma_inc_vec_2 = gamma_lower(
            1 + n_vec, -k * self.lambda_p * np.ones_like(n_vec)
        )
        serie3 = (
            taylor_term
            * gamma(2 * self.alpha_ubar - 1 - n_vec)
            * one_minus_alphap_plus_n
            * self.lambda_ubar ** (1 + n_vec - self.alpha_ubar)
            * self.lambda_p ** (-1 - n_vec)
            * gamma_inc_vec_2
        ).sum() * self.mbg

        return serie1_1 + serie1_2 - serie2 - serie3

    def serie_AN(
        self,
        k: float | npt.NDArray[np.float64],
        ttm: float | npt.NDArray[np.float64],
        N: int,
    ):
        n_vec = np.arange(0, N)
        fact_n = factorial(n_vec)
        taylor_term = (-1) ** n_vec / fact_n
        # doublon
        am_plus_m = gamma(self.alpha_m + n_vec)
        one_minus_alphap_plus_n = gamma(1 - self.alpha_p + n_vec)

        serie1_1 = np.exp(-self.zeta)

        num = (self.lambda_p**self.alpha_p * self.lambda_m**self.alpha_m) * gamma(
            2 * self.alpha_ubar
        )
        denum = -(
            self.lambda_ubar ** (2 * self.alpha_ubar)
            * gamma(self.alpha_p)
            * gamma(self.alpha_m)
            * (self.alpha_p)
        )
        serie1_2 = (num / denum) * hyp2f1(
            2 * self.alpha_ubar,
            1,
            1 + self.alpha_p,
            (self.lambda_p - 1) / self.lambda_ubar,
        )

        gamma_inc_vec = np.array(
            gamma_lower(
                n_vec + 2 * self.alpha_ubar,
                -k * (self.lambda_p - 1) * np.ones_like(n_vec),
            )
        )

        serie2 = (
            taylor_term
            * gamma(1 - n_vec - 2 * self.alpha_ubar)
            * am_plus_m
            * self.lambda_ubar ** (self.alpha_ubar + n_vec)
            * (self.lambda_p - 1) ** (-n_vec - 2 * self.alpha_ubar)
            * gamma_inc_vec
        ).sum() * self.mbg

        gamma_inc_vec_2 = gamma_lower(
            1 + n_vec, -k * (self.lambda_p - 1) * np.ones_like(n_vec)
        )
        serie3 = (
            taylor_term
            * gamma(2 * self.alpha_ubar - 1 - n_vec)
            * one_minus_alphap_plus_n
            * self.lambda_ubar ** (1 + n_vec - self.alpha_ubar)
            * (self.lambda_p - 1) ** (-1 - n_vec)
            * gamma_inc_vec_2
        ).sum() * self.mbg

        return serie1_1 + serie1_2 - serie2 - serie3

    def serie_EUR_1(
        self,
        k: float | npt.NDArray[np.float64],
        ttm: float | npt.NDArray[np.float64],
        N: int,
    ):
        # faire attention AN => Ke^{k-RT}
        # CN => e^{-RT}
        # ici, je fais l'erreur

        n_vec = np.arange(0, N)
        fact_n = factorial(n_vec)
        taylor_term = (-1) ** n_vec / fact_n
        # doublons
        am_plus_m = gamma(self.alpha_m + n_vec)
        two_alpha_bar_plus_n = gamma(n_vec + 2 * self.alpha_ubar)
        one_minus_alphap_plus_n = gamma(1 - self.alpha_p + n_vec)

        serie1_1 = (
            taylor_term
            * am_plus_m
            * one_minus_alphap_plus_n
            * gamma(self.alpha_p - n_vec)
            * self.lambda_ubar ** (self.alpha_bar - n_vec)
            * (
                np.exp(k) * (self.lambda_p - 1) ** (n_vec - self.alpha_p)
                - (self.lambda_p) ** (n_vec - self.alpha_p)
            )
        ).sum()

        serie1_2 = (
            taylor_term
            * two_alpha_bar_plus_n
            * fact_n
            * gamma(-self.alpha_p - n_vec)
            * self.lambda_ubar ** (-self.alpha_ubar - n_vec)
            * (np.exp(k) * (self.lambda_p - 1) ** (n_vec) - (self.lambda_p) ** (n_vec))
        ).sum()

        gamma_inc_vec_lamp = np.array(
            gamma_lower(
                n_vec + 2 * self.alpha_ubar,
                -k * self.lambda_p * np.ones_like(n_vec),
            )
        )

        gamma_inc_vec_lamp_m1 = np.array(
            gamma_lower(
                n_vec + 2 * self.alpha_ubar,
                -k * (self.lambda_p - 1) * np.ones_like(n_vec),
            )
        )

        serie2 = (
            taylor_term
            * gamma(1 - n_vec - 2 * self.alpha_ubar)
            * am_plus_m
            * self.lambda_ubar ** (self.alpha_ubar + n_vec)
            * (
                np.exp(k)
                * (self.lambda_p - 1) ** (-n_vec - 2 * self.alpha_ubar)
                * gamma_inc_vec_lamp_m1
                - self.lambda_p ** (-n_vec - 2 * self.alpha_ubar) * gamma_inc_vec_lamp
            )
        ).sum()

        gamma_inc_vec_2_lamp = gamma_lower(
            1 + n_vec, -k * self.lambda_p * np.ones_like(n_vec)
        )
        gamma_inc_vec_2_lamp_m1 = gamma_lower(
            1 + n_vec, -k * (self.lambda_p - 1) * np.ones_like(n_vec)
        )
        serie3 = (
            taylor_term
            * gamma(2 * self.alpha_ubar - 1 - n_vec)
            * one_minus_alphap_plus_n
            * self.lambda_ubar ** (1 + n_vec - self.alpha_ubar)
            * (
                np.exp(k)
                * (self.lambda_p - 1) ** (-1 - n_vec)
                * gamma_inc_vec_2_lamp_m1
                - self.lambda_p ** (-1 - n_vec) * gamma_inc_vec_2_lamp
            )
        ).sum()

        serie1 = serie1_1 + serie1_2
        print("serie EUR")
        return serie1 - serie2 - serie3

    def price_eur(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        ttm: float,
        N: int = 25,
        # time_verbose=True,
    ):
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        if k > 0:
            raise NotImplementedError("Negative moneyness not implemented so far.")
        elif k == 0:
            raise NotImplementedError("Negative moneyness not implemented so far.")
        else:
            serie = self.serie_EUR_1(k, ttm, N)
            call_price = self.mbg * K * np.exp(-r * ttm) * serie
            return call_price
