"""BGPricer class"""

import warnings

import numpy as np
import numpy.typing as npt
import scipy.special as sc

from src.gamma_func_cpp.gamma_lower import (
    gamma_lower as gamma_lower_cpp,
)


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

        # self.mbg = self.get_mbg()
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
        return -alpha * sc.gamma(-beta)

    def zeta_(self):
        """convexity adjustment

        Returns:
            zeta: convex. adj
        """
        zeta_p = ((self.lambda_p / (self.lambda_p - 1)) ** self.alpha_p) * (
            (self.lambda_m / (self.lambda_m + 1)) ** self.alpha_m
        )
        return -np.log(zeta_p)

    def get_mbg(self, ttm: float):
        """TBD"""
        # density constant BG
        numerator = (self.lambda_p**(self.alpha_p*ttm)) * \
            (self.lambda_m**(self.alpha_m*ttm))
        denominator = (
            self.lambda_ubar ** (ttm*self.alpha_ubar)
            * sc.gamma((self.alpha_p*ttm))
            * sc.gamma((self.alpha_m*ttm))
            * sc.gamma(1 - (self.alpha_p*ttm))
        )
        Mbg = numerator / denominator
        return Mbg

    def serie_eur(
        self,
        k: float | npt.NDArray[np.float64],
        ttm: float | npt.NDArray[np.float64],
        N: int,
    ):
        """European option pricing with Mellin series"""

        # BG model constants
        alpha_p_t = self.alpha_p*ttm
        alpha_m_t = self.alpha_m*ttm
        alpha_ubar_t = self.alpha_ubar*ttm
        alpha_bar_t = self.alpha_bar*ttm

        n_vec = np.arange(0, N)
        fact_n = sc.factorial(n_vec)
        taylor_term = (-1) ** n_vec / fact_n
        am_plus_m = sc.gamma(alpha_m_t + n_vec)
        two_alpha_bar_plus_n = sc.gamma(n_vec + 2 * alpha_ubar_t)
        one_minus_alphap_plus_n = sc.gamma(1 - alpha_p_t + n_vec)

        serie1_1 = (
            taylor_term
            * am_plus_m
            * one_minus_alphap_plus_n
            * sc.gamma(alpha_p_t - n_vec)
            * self.lambda_ubar ** (alpha_bar_t - n_vec)
            * (
                np.exp(k) * (self.lambda_p - 1) ** (n_vec - alpha_p_t)
                - (self.lambda_p) ** (n_vec - alpha_p_t)
            )
        ).sum()

        serie1_2 = (
            taylor_term
            * two_alpha_bar_plus_n
            * fact_n
            * sc.gamma(-alpha_p_t - n_vec)
            * self.lambda_ubar ** (-alpha_ubar_t - n_vec)
            * (np.exp(k) * (self.lambda_p - 1) ** (n_vec) - (self.lambda_p) ** (n_vec))
        ).sum()

        gamma_inc_vec_lamp = np.array(
            gamma_lower_cpp(
                n_vec + 2 * alpha_ubar_t,
                -k * self.lambda_p,
            )
        )

        gamma_inc_vec_lamp_m1 = np.array(
            gamma_lower_cpp(
                n_vec + 2 * alpha_ubar_t,
                -k * (self.lambda_p - 1),
            )
        )

        serie2 = (
            taylor_term
            * sc.gamma(1 - n_vec - 2 * alpha_ubar_t)
            * am_plus_m
            * self.lambda_ubar ** (alpha_ubar_t + n_vec)
            * (
                np.exp(k)
                * (self.lambda_p - 1) ** (-n_vec - 2 * alpha_ubar_t)
                * gamma_inc_vec_lamp_m1
                - self.lambda_p ** (-n_vec - 2 *
                                    alpha_ubar_t) * gamma_inc_vec_lamp
            )
        ).sum()

        gamma_inc_vec_2_lamp = gamma_lower_cpp(
            1 + n_vec, -k * self.lambda_p
        )
        gamma_inc_vec_2_lamp_m1 = gamma_lower_cpp(
            1 + n_vec, -k * (self.lambda_p - 1)
        )
        serie3 = (
            taylor_term
            * sc.gamma(2 * alpha_ubar_t - 1 - n_vec)
            * one_minus_alphap_plus_n
            * self.lambda_ubar ** (1 + n_vec - alpha_ubar_t)
            * (
                np.exp(k)
                * (self.lambda_p - 1) ** (-1 - n_vec)
                * gamma_inc_vec_2_lamp_m1
                - self.lambda_p ** (-1 - n_vec) * gamma_inc_vec_2_lamp
            )
        ).sum()

        serie1 = serie1_1 + serie1_2
        return serie1 - serie2 - serie3

    def price(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        ttm: float,
        N: int = 25,
        # time_verbose=True,
    ):
        """price function"""
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        if k > 0:
            raise NotImplementedError(
                "Negative moneyness not implemented so far.")
        elif k == 0:
            raise NotImplementedError(
                "Negative moneyness not implemented so far.")
        else:
            serie = self.serie_eur(k, ttm, N)
            call_price = self.get_mbg(ttm) * K * np.exp(-r * ttm) * serie
            return call_price

    #######################
    #######################
    ## Cash or nothing ####
    #######################
    #######################

    # def price_cn(
    #     self,
    #     S0: float,
    #     K: float,
    #     r: float,
    #     q: float,
    #     ttm: float,
    #     N: int = 25,
    #     # time_verbose=True,
    # ):
    #     """cn"""
    #     k = np.log(S0 / K) + (r - q + self.zeta) * ttm
    #     serie = self.serie_CN(k, ttm, N)
    #     return np.exp(-r * ttm) * serie

    ###########################
    ###########################
    ## Eur by diff (AN/CN) ####
    ###########################
    ###########################

    # def price_eur_diff(
    #     self,
    #     S0: float,
    #     K: float,
    #     r: float,
    #     q: float,
    #     ttm: float,
    #     N: int = 25,
    #     # time_verbose=True,
    # ):
    #     """TBD"""
    #     k = np.log(S0 / K) + (r - q + self.zeta) * ttm
    #     CN_value = self.serie_CN(k, ttm, N)
    #     AN_value = np.exp(k) * self.serie_AN(k, ttm, N)

    #     return K * np.exp(-r * ttm) * (AN_value - CN_value)

    # def serie_CN(
    #     self,
    #     k: float | npt.NDArray[np.float64],
    #     ttm: float | npt.NDArray[np.float64],
    #     N: int,
    # ):
    #     """TBD"""
    #     n_vec = np.arange(0, N)
    #     fact_n = sc.factorial(n_vec)
    #     taylor_term = (-1) ** n_vec / fact_n
    #     # doublon
    #     am_plus_m = sc.gamma(self.alpha_m + n_vec)
    #     one_minus_alphap_plus_n = sc.gamma(1 - self.alpha_p + n_vec)

    #     serie1_1 = 1

    #     num = (self.lambda_p**self.alpha_p * self.lambda_m**self.alpha_m) * sc.gamma(
    #         2 * self.alpha_ubar
    #     )
    #     denum = -(
    #         self.lambda_ubar ** (2 * self.alpha_ubar)
    #         * sc.gamma(self.alpha_p)
    #         * sc.gamma(self.alpha_m)
    #         * (self.alpha_p)
    #     )
    #     # must disable since all unfunc are not found by pylint
    #     # pylint: disable=E1101
    #     serie1_2 = (num / denum) * sc.hyp2f1(
    #         2 * self.alpha_ubar,
    #         1,
    #         1 + self.alpha_p,
    #         self.lambda_p / self.lambda_ubar,
    #     )
    #     # pylint: enable=E1101

    #     gamma_inc_vec = gamma_lower_cpp(
    #         n_vec + 2 * self.alpha_ubar, -k * self.lambda_p)

    #     serie2 = (
    #         taylor_term
    #         * sc.gamma(1 - n_vec - 2 * self.alpha_ubar)
    #         * am_plus_m
    #         * self.lambda_ubar ** (self.alpha_ubar + n_vec)
    #         * self.lambda_p ** (-n_vec - 2 * self.alpha_ubar)
    #         * gamma_inc_vec
    #     ).sum() * self.mbg

    #     gamma_inc_vec_2 = gamma_lower_cpp(1 + n_vec, -k * self.lambda_p)
    #     serie3 = (
    #         taylor_term
    #         * sc.gamma(2 * self.alpha_ubar - 1 - n_vec)
    #         * one_minus_alphap_plus_n
    #         * self.lambda_ubar ** (1 + n_vec - self.alpha_ubar)
    #         * self.lambda_p ** (-1 - n_vec)
    #         * gamma_inc_vec_2
    #     ).sum() * self.mbg

    #     return serie1_1 + serie1_2 - serie2 - serie3

    # def serie_AN(
    #     self,
    #     k: float | npt.NDArray[np.float64],
    #     ttm: float | npt.NDArray[np.float64],
    #     N: int,
    # ):
    #     """TBD"""
    #     n_vec = np.arange(0, N)
    #     fact_n = sc.factorial(n_vec)
    #     taylor_term = (-1) ** n_vec / fact_n
    #     # doublon
    #     am_plus_m = sc.gamma(self.alpha_m + n_vec)
    #     one_minus_alphap_plus_n = sc.gamma(1 - self.alpha_p + n_vec)

    #     serie1_1 = np.exp(-self.zeta)

    #     num = (self.lambda_p**self.alpha_p * self.lambda_m**self.alpha_m) * sc.gamma(
    #         2 * self.alpha_ubar
    #     )
    #     denum = -(
    #         self.lambda_ubar ** (2 * self.alpha_ubar)
    #         * sc.gamma(self.alpha_p)
    #         * sc.gamma(self.alpha_m)
    #         * (self.alpha_p)
    #     )
    #     # pylint: disable=E1101
    #     serie1_2 = (num / denum) * sc.hyp2f1(
    #         2 * self.alpha_ubar,
    #         1,
    #         1 + self.alpha_p,
    #         (self.lambda_p - 1) / self.lambda_ubar,
    #     )
    #     # pylint: enable=E1101

    #     gamma_inc_vec = np.array(
    #         gamma_lower_cpp(
    #             n_vec + 2 * self.alpha_ubar,
    #             -k * (self.lambda_p - 1),
    #         )
    #     )

    #     serie2 = (
    #         taylor_term
    #         * sc.gamma(1 - n_vec - 2 * self.alpha_ubar)
    #         * am_plus_m
    #         * self.lambda_ubar ** (self.alpha_ubar + n_vec)
    #         * (self.lambda_p - 1) ** (-n_vec - 2 * self.alpha_ubar)
    #         * gamma_inc_vec
    #     ).sum() * self.mbg

    #     gamma_inc_vec_2 = gamma_lower_cpp(
    #         1 + n_vec, -k * (self.lambda_p - 1)
    #     )
    #     serie3 = (
    #         taylor_term
    #         * sc.gamma(2 * self.alpha_ubar - 1 - n_vec)
    #         * one_minus_alphap_plus_n
    #         * self.lambda_ubar ** (1 + n_vec - self.alpha_ubar)
    #         * (self.lambda_p - 1) ** (-1 - n_vec)
    #         * gamma_inc_vec_2
    #     ).sum() * self.mbg

    #     return serie1_1 + serie1_2 - serie2 - serie3
