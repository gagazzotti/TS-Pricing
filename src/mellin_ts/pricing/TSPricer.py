"""Importing modules"""

import warnings
from time import time

import numpy as np
import numpy.typing as npt
import scipy
import scipy.special
import scipy.special as sc
from scipy.special import gamma, poch

np.set_printoptions(precision=16)

# pylint: disable=all

from src.mellin_ts.pricing.lower_gamma_vect.gamma_incomp import (
    gamma_lower_incomplete_non_normalized,
)

# pylint: enable=all


warnings.filterwarnings("ignore", category=RuntimeWarning)

# TODO: faire un fichier de test gamma lower, faire un fichier d'import python de


def gamma_lower_cpp(a, z):
    return gamma_lower_incomplete_non_normalized(a, z)


class TemperedStablePricer:
    """
    Tempered Stable Pricer Pricer
    """

    def __init__(
        self,
        alpha_p: float,
        beta_p: float,
        lambda_p: float,
        alpha_m: float,
        beta_m: float,
        lambda_m: float,
    ):
        # raw parameters
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.lambda_p = lambda_p
        self.alpha_m = alpha_m
        self.beta_m = beta_m
        self.lambda_m = lambda_m
        # transformed parameters
        self.ap = -alpha_p * sc.gamma(-beta_p)
        self.am = -alpha_m * sc.gamma(-beta_m)
        self.ulambda = lambda_m + lambda_p
        self.gamma = self.get_gamma()
        # convexity adjustment
        self.zeta = self.get_zeta()
        return

    def get_gamma(self) -> float:
        """gamma constant in the paper

        Returns:
            float: gamma
        """
        return (
            self.ap * self.lambda_p**self.beta_p + self.am * self.lambda_m**self.beta_m
        )

    def get_zeta(self):
        """convexity adjustment

        Returns:
            zeta: convex. adj
        """
        zeta_p = (
            self.alpha_p
            * sc.gamma(-self.beta_p)
            * ((self.lambda_p - 1) ** self.beta_p - self.lambda_p**self.beta_p)
        )
        zeta_m = (
            self.alpha_m
            * sc.gamma(-self.beta_m)
            * ((self.lambda_m + 1) ** self.beta_m - self.lambda_m**self.beta_m)
        )
        return -(zeta_p + zeta_m)

    def a2_3_indexes(self, N: int, ttm, k):
        # ttm_vec = np.array(ttm)[None, None, None, :, None]
        n1 = np.arange(N)[:, None, None]
        n2 = np.arange(N)[None, :, None]
        n3 = np.arange(N)[None, None, :]
        taylor = (-1) ** (n1 + n2 + n3) / (
            sc.factorial(n1) * sc.factorial(n2) * sc.factorial(n3)
        )
        pochhamer_symb = poch(1 + self.beta_p * n2, n1)
        gamma_term = sc.gamma(-1 - n1 - self.beta_p * n2 - self.beta_m * n3) / (
            sc.gamma(-self.beta_p * n2) * sc.gamma(-self.beta_m * n3)
        )
        gamma_term[:, 0, 0] = 0
        # time term
        at_term = (self.ap * ttm) ** n2 * (self.am * ttm) ** n3
        a2 = taylor * pochhamer_symb * gamma_term * at_term
        ulambda_term = self.ulambda ** (1 + n1 + self.beta_p * n2 + self.beta_m * n3)
        k_vec = np.ones_like(n1).astype(float) * k.item()
        function_term = (
            np.exp(k_vec)
            * (self.lambda_p - 1) ** (-1 - n1)
            * (gamma_lower_cpp(1 + n1, -(self.lambda_p - 1) * k_vec))
        ) - (
            (self.lambda_p) ** (-1 - n1)
            * (gamma_lower_cpp(1 + n1, -(self.lambda_p) * k_vec))
        )
        full_a2 = -a2 * function_term * ulambda_term
        return full_a2

    def a1_3_indexes(self, N: int, ttm, k):
        n1 = np.arange(N)[:, None, None]
        n2 = np.arange(N)[None, :, None]
        n3 = np.arange(N)[None, None, :]
        taylor = (-1) ** (n1 + n2 + n3) / (
            sc.factorial(n1) * sc.factorial(n2) * sc.factorial(n3)
        )
        pochhamer_symb = poch(-self.beta_m * n3, n1) / sc.gamma(1 + self.beta_p * n2)
        at_term = (self.ap * ttm) ** n2 * (self.am * ttm) ** n3
        ulambda_term = self.ulambda ** (n1)
        # A1 pas encore vectorisé, on pourra réduire les dimensions une fois A1 fait
        k_vec = np.ones_like(n1 + n2 + n3).astype(float) * float(k)
        # piecwise
        low_gamma_term = np.zeros_like(n1 + n2 + n3).astype(float)
        # gamma_term in 0,0,0
        # other
        low_gamma_term = sc.gamma(1 - n1 + self.beta_p * n2 + self.beta_m * n3) / (
            sc.gamma(-self.beta_p * n2)
        )
        low_gamma_term = low_gamma_term * (
            (
                np.exp(k_vec)
                * (self.lambda_p - 1) ** (-n1 + self.beta_p * n2 + self.beta_m * n3)
                * np.array(
                    gamma_lower_cpp(
                        n1 - self.beta_p * n2 - self.beta_m * n3,
                        -(self.lambda_p - 1) * k_vec,
                    )
                )
            )
            - (
                (self.lambda_p) ** (-n1 + self.beta_p * n2 + self.beta_m * n3)
                * np.array(
                    gamma_lower_cpp(
                        n1 - self.beta_p * n2 - self.beta_m * n3,
                        -(self.lambda_p) * k_vec,
                    )
                )
            )
        ).astype(float)

        # multiplication par e^{k}-1
        low_gamma_term[0, 0, 0] = (
            self.beta_p / (self.beta_m + self.beta_p) * (np.exp(k) - 1).astype(float)
        )

        # gamma_term in >1,0,0
        low_gamma_term[1:, 0, 0] = 0
        a1 = taylor * pochhamer_symb * at_term
        full_a1 = -a1 * ulambda_term * low_gamma_term
        return full_a1

    def serie1(self, k: float, ttm: float, N: int):
        term1_3_index = self.a1_3_indexes(N, ttm, k)
        serie1 = (term1_3_index).sum(axis=(0, 1, 2))
        term2_3_index = self.a2_3_indexes(N, ttm, k)
        serie2 = (term2_3_index).sum(axis=(0, 1, 2))
        serie = serie1 + serie2
        return serie

    def serie2(self, k: float, ttm: float, N: int):
        k_vec = float(k)
        n1 = np.arange(N)[:, None, None]
        n2 = np.arange(N)[None, :, None]
        n3 = np.arange(N)[None, None, :]
        taylor_term = -((-1) ** (n2 + n3)) / (sc.factorial(n2) * sc.factorial(n3))
        gamma_term = np.zeros((N, N, N))
        gamma_term = sc.gamma(-self.beta_p * n2 - self.beta_m * n3 + n1) / (
            sc.gamma(1 - self.beta_p * n2 + n1) * sc.gamma(-self.beta_m * n3)
        )
        gamma_term[0, 0, 0] = self.beta_m / (self.beta_m + self.beta_p)
        ulambda_term = self.ulambda ** (self.beta_p * n2 + self.beta_m * n3 - n1)
        exp_term = np.exp(k_vec) * (self.lambda_p - 1) ** n1 - self.lambda_p**n1
        at = (self.ap * ttm) ** n2 * (self.am * ttm) ** n3
        serie = (taylor_term * gamma_term * exp_term * at * ulambda_term).sum(
            axis=(0, 1, 2)
        )
        return serie

    def price(
        self,
        S0: float,
        K: float | npt.NDArray[np.float64],
        r: float,
        q: float,
        ttm: float | npt.NDArray[np.float64],
        N: int = 5,
    ):
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        if k > 0:
            raise NotImplementedError("Negative moneyness not implemented so far.")
        elif k == 0:
            raise NotImplementedError("Negative moneyness not implemented so far.")
        else:
            serie1 = self.serie1(k, ttm, N)
            serie2 = self.serie2(k, ttm, N)
            constant_term = np.exp(k - self.zeta * ttm) - 1
            factor_serie = np.exp(self.gamma * ttm)
            factor = K * np.exp(-r * ttm)
            call_price = factor * (constant_term + factor_serie * (serie1 + serie2))
            return float(call_price)
