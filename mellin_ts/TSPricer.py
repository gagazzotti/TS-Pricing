"""Importing modules"""

import time
import tqdm
import mpmath
import itertools as it
import numpy as np
import numpy.typing as npt
from scipy.special import gamma, factorial, poch
import warnings
from mellin_ts.upper_gamma.gamma_module import gamma_upper_incomplete


warnings.filterwarnings("ignore", category=RuntimeWarning)


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
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.lambda_p = lambda_p
        self.alpha_m = alpha_m
        self.beta_m = beta_m
        self.lambda_m = lambda_m
        self.ap = self.a(alpha_p, beta_p)
        self.am = self.a(alpha_m, beta_m)
        self.zeta = self.zeta_()
        self.ulambda = lambda_m + lambda_p
        self.gamma = self.gamma_()

        return

    def display_params(self, k: float):
        """Display params in the terminal

        Args:
            k (float): moneyness
        """
        print(f"Moneyness (k): {k}")
        print(f"zeta: {self.zeta}")
        print(f"gamma: {self.gamma}")

    def a(self, alpha: float, beta: float) -> float:
        """a_pm constant in the paper

        Args:
            alpha (float): alpha
            beta (float): beta

        Returns:
            float: apm
        """
        return -alpha * gamma(-beta)

    def gamma_(self) -> float:
        """gamma constant in the paper

        Returns:
            float: gamma
        """
        return (
            self.ap * self.lambda_p**self.beta_p + self.am * self.lambda_m**self.beta_m
        )

    def zeta_(self):
        """convexity adjustment

        Returns:
            zeta: convex. adj
        """
        zeta_p = (
            self.alpha_p
            * gamma(-self.beta_p)
            * ((self.lambda_p - 1) ** self.beta_p - self.lambda_p**self.beta_p)
        )
        zeta_m = (
            self.alpha_m
            * gamma(-self.beta_m)
            * ((self.lambda_m + 1) ** self.beta_m - self.lambda_m**self.beta_m)
        )
        return -(zeta_p + zeta_m)

    def serie2(self, k: float, ttm: float, N: int):
        n1 = np.arange(N)[:, None, None]
        n2 = np.arange(N)[None, :, None]
        n3 = np.arange(N)[None, None, :]
        taylor_term = -((-1) ** (n1 + n2)) / (factorial(n1) * factorial(n2))
        gamma_term = np.zeros((N, N, N))
        gamma_term = gamma(-self.beta_p * n1 - self.beta_m * n2 + n3) / (
            gamma(1 - self.beta_p * n1 + n3) * gamma(-self.beta_m * n2)
        )
        gamma_term[0, 0, 0] = self.beta_m / (self.beta_m + self.beta_p)
        ulambda_term = self.ulambda ** (self.beta_p * n1 + self.beta_m * n2 - n3)
        exp_term = np.exp(k) * (self.lambda_p - 1) ** n3 - self.lambda_p**n3
        at = (self.ap * ttm) ** n1 * (self.am * ttm) ** n2
        serie = (taylor_term * gamma_term * exp_term * at * ulambda_term).sum()
        return serie

    def serie1(self, k: float, ttm: float, N: int):
        n1 = np.arange(N)[:, None, None, None]
        n2 = np.arange(N)[None, :, None, None]
        n3 = np.arange(N)[None, None, :, None]
        n4 = np.arange(N)[None, None, None, :]

        term1 = (
            self.a1(N, ttm)
            * self.ulambda**n1
            * (-k) ** (n1 - self.beta_p * n2 - self.beta_m * n3 + n4)
        )
        # print(term1.shape)
        term2 = (
            self.a2(N, ttm)
            * self.ulambda ** (1 + n1 + self.beta_p * n2 + self.beta_m * n3)
            * (-k) ** (1 + n1 + n4)
        )
        exp_term = np.exp(k) * (self.lambda_p - 1) ** n4 - self.lambda_p**n4
        serie = (term1 + term2) * exp_term
        serie = serie.sum()
        return serie

    def a1(self, N: int, ttm):
        # ttm = np.array(ttm)[None, None, None, None, :]
        n1 = np.arange(N)[:, None, None, None]
        n2 = np.arange(N)[None, :, None, None]
        n3 = np.arange(N)[None, None, :, None]
        n4 = np.arange(N)[None, None, None, :]
        taylor = (-1) ** (n1 + n2 + n3) / (
            factorial(n1) * factorial(n2) * factorial(n3) * factorial(n4)
        )
        pochhamer_symb = (
            poch(-self.beta_m * n3, n1)
            * poch(n1 - self.beta_p * n2 - self.beta_m * n3, n4)
            / gamma(1 + self.beta_p * n2)
        )
        at_term = (self.ap * ttm) ** n2 * (self.am * ttm) ** n3
        # print(at_term.shape)
        gamma_term = gamma(-n1 + self.beta_p * n2 + self.beta_m * n3 - n4) / (
            gamma(-self.beta_p * n2)
        )
        gamma_term[:, 0, 0, :] = (
            ((-1) ** (1 + n1 + n4) / factorial(n1 + n4))
            * (self.beta_p / (self.beta_p + self.beta_m))
        )[:, 0, 0, :]
        a1 = pochhamer_symb * at_term * gamma_term * taylor
        # print(a1.sum())

        # print("###########################")
        return a1

    def a2(self, N: int, ttm):
        n1 = np.arange(N)[:, None, None, None]
        n2 = np.arange(N)[None, :, None, None]
        n3 = np.arange(N)[None, None, :, None]
        n4 = np.arange(N)[None, None, None, :]
        taylor = (-1) ** (n1 + n2 + n3 + n4) / (
            factorial(n1) * factorial(n2) * factorial(n3) * factorial(n4)
        )
        pochhamer_symb = poch(1 + self.beta_p * n2, n1) / (-1 - n1 - n4)
        gamma_term = gamma(-1 - n1 - self.beta_p * n2 - self.beta_m * n3) / (
            gamma(-self.beta_p * n2) * gamma(-self.beta_m * n3)
        )
        gamma_term[:, 0, 0, :] = 0
        # time term
        at_term = (self.ap * ttm) ** n2 * (self.am * ttm) ** n3
        a2 = taylor * pochhamer_symb * gamma_term * at_term
        return a2

    def price(self, S0: float, K: float, r: float, q: float, ttm: float, N: int = 5):
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        serie1 = self.serie1(k, ttm, N)
        serie2 = self.serie2(k, ttm, N)
        constant_term = np.exp(k - self.zeta * ttm) - 1
        # factors
        factor_serie = np.exp(self.gamma * ttm)
        factor = K * np.exp(-r * ttm)
        call_price = factor * (constant_term + factor_serie * (serie1 + serie2))
        return call_price

    #####################
    #### VECT ###########
    #####################

    def a1_vect(self, N: int, ttm):
        ttm_vec = np.array(ttm)[None, None, None, None, :]
        n1 = np.arange(N)[:, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None]
        n4 = np.arange(N)[None, None, None, :, None]
        taylor = (-1) ** (n1 + n2 + n3) / (
            factorial(n1) * factorial(n2) * factorial(n3) * factorial(n4)
        )
        pochhamer_symb = (
            poch(-self.beta_m * n3, n1)
            * poch(n1 - self.beta_p * n2 - self.beta_m * n3, n4)
            / gamma(1 + self.beta_p * n2)
        )
        at_term = (self.ap * ttm_vec) ** n2 * (self.am * ttm_vec) ** n3
        # print(at_term.shape)
        gamma_term = gamma(-n1 + self.beta_p * n2 + self.beta_m * n3 - n4) / (
            gamma(-self.beta_p * n2)
        )
        gamma_term[:, 0, 0, :, :] = (
            ((-1) ** (1 + n1 + n4) / factorial(n1 + n4))
            * (self.beta_p / (self.beta_p + self.beta_m))
        )[:, 0, 0, :, :]
        a1 = pochhamer_symb * at_term * gamma_term * taylor
        # print(a1.sum())
        # print(a1.shape)
        print("###########################")
        return a1

    def a2_vect(self, N: int, ttm):
        ttm_vec = np.array(ttm)[None, None, None, None, :]
        n1 = np.arange(N)[:, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None]
        n4 = np.arange(N)[None, None, None, :, None]
        taylor = (-1) ** (n1 + n2 + n3 + n4) / (
            factorial(n1) * factorial(n2) * factorial(n3) * factorial(n4)
        )
        pochhamer_symb = poch(1 + self.beta_p * n2, n1) / (-1 - n1 - n4)
        gamma_term = gamma(-1 - n1 - self.beta_p * n2 - self.beta_m * n3) / (
            gamma(-self.beta_p * n2) * gamma(-self.beta_m * n3)
        )
        gamma_term[:, 0, 0, :] = 0
        # time term
        at_term = (self.ap * ttm_vec) ** n2 * (self.am * ttm_vec) ** n3
        a2 = taylor * pochhamer_symb * gamma_term * at_term
        return a2

    def serie1_vect(self, k: float, ttm: float, N: int):
        ttm_vec = np.array(ttm)[None, None, None, None, :]
        n1 = np.arange(N)[:, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None]
        n4 = np.arange(N)[None, None, None, :, None]

        term1 = (
            self.a1_vect(N, ttm)
            * self.ulambda**n1
            * (-k) ** (n1 - self.beta_p * n2 - self.beta_m * n3 + n4)
        )
        term2 = (
            self.a2_vect(N, ttm)
            * self.ulambda ** (1 + n1 + self.beta_p * n2 + self.beta_m * n3)
            * (-k) ** (1 + n1 + n4)
        )
        exp_term = np.exp(k) * (self.lambda_p - 1) ** n4 - self.lambda_p**n4
        serie = (term1 + term2) * exp_term
        serie = serie.sum(axis=(0, 1, 2, 3))
        return serie

    def serie2_vect(self, k: float, ttm: float, N: int):
        k_vec = k[0]
        print(k_vec.shape, k.shape)
        ttm_vec = np.array(ttm)[None, None, None, :]
        n1 = np.arange(N)[:, None, None, None]
        n2 = np.arange(N)[None, :, None, None]
        n3 = np.arange(N)[None, None, :, None]
        taylor_term = -((-1) ** (n1 + n2)) / (factorial(n1) * factorial(n2))
        gamma_term = np.zeros((N, N, N))
        gamma_term = gamma(-self.beta_p * n1 - self.beta_m * n2 + n3) / (
            gamma(1 - self.beta_p * n1 + n3) * gamma(-self.beta_m * n2)
        )
        gamma_term[0, 0, 0] = self.beta_m / (self.beta_m + self.beta_p)
        ulambda_term = self.ulambda ** (self.beta_p * n1 + self.beta_m * n2 - n3)
        exp_term = np.exp(k_vec) * (self.lambda_p - 1) ** n3 - self.lambda_p**n3
        at = (self.ap * ttm_vec) ** n1 * (self.am * ttm_vec) ** n2
        serie = (taylor_term * gamma_term * exp_term * at * ulambda_term).sum(
            axis=(0, 1, 2)
        )
        return serie

    def price_vect(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        ttm: npt.NDArray[np.float64],
        N: int = 5,
    ):
        ttm_vec = np.array(ttm)[None, None, None, None, :]
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm_vec
        serie1 = self.serie1_vect(k, ttm, N)
        serie2 = self.serie2_vect(k, ttm, N)
        constant_term = np.exp(k - self.zeta * ttm) - 1
        # # factors
        factor_serie = np.exp(self.gamma * ttm)
        factor = K * np.exp(-r * ttm)
        call_price = factor * (constant_term + factor_serie * (serie1 + serie2))
        return call_price
