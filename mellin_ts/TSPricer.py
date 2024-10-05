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

    def a1_vect(self, N: int, ttm):
        ttm_vec = np.array(ttm)[None, None, None, None, :, None]
        n1 = np.arange(N)[:, None, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None, None]
        n4 = np.arange(N)[None, None, None, :, None, None]
        taylor = (-1) ** (n1 + n2 + n3) / (
            factorial(n1) * factorial(n2) * factorial(n3) * factorial(n4)
        )
        pochhamer_symb = (
            poch(-self.beta_m * n3, n1)
            * poch(n1 - self.beta_p * n2 - self.beta_m * n3, n4)
            / gamma(1 + self.beta_p * n2)
        )
        at_term = (self.ap * ttm_vec) ** n2 * (self.am * ttm_vec) ** n3
        # print("at", at_term.shape)
        gamma_term = gamma(-n1 + self.beta_p * n2 + self.beta_m * n3 - n4) / (
            gamma(-self.beta_p * n2)
        )
        gamma_term[:, 0, 0, :, :] = (
            ((-1) ** (1 + n1 + n4) / factorial(n1 + n4))
            * (self.beta_p / (self.beta_p + self.beta_m))
        )[:, 0, 0, :, :]
        a1 = pochhamer_symb * at_term * gamma_term * taylor
        return a1

    def a2_vect(self, N: int, ttm):
        ttm_vec = np.array(ttm)[None, None, None, :, None]
        n1 = np.arange(N)[:, None, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None, None]
        n4 = np.arange(N)[None, None, None, :, None, None]
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
        ttm_vec = np.array(ttm)[None, None, None, :, None]
        n1 = np.arange(N)[:, None, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None, None]
        n4 = np.arange(N)[None, None, None, :, None, None]

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
        ttm_vec = np.array(ttm)[None, None, None, :, None]
        n1 = np.arange(N)[:, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None]
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

    def price(
        self,
        S0: float,
        K: float | npt.NDArray[np.float64],
        r: float,
        q: float,
        ttm: float | npt.NDArray[np.float64],
        N: int = 5,
    ):
        single_strike = False
        single_ttm = False
        if isinstance(K, float) or isinstance(K, int):
            single_strike = True
            K_vec = np.array([K])[None, None, None, None, None, :]
        else:
            K_vec = np.array(K)[None, None, None, None, None, :]

        if isinstance(ttm, float) or isinstance(ttm, int):
            single_ttm = True
            ttm_vec = np.array([ttm])[None, None, None, None, :, None]
            ttm_alone = np.array([ttm])
        else:
            ttm_vec = np.array(ttm)[None, None, None, None, :, None]
            ttm_alone = ttm
        # print(ttm_vec.shape)
        k = np.log(S0 / K_vec) + (r - q + self.zeta) * ttm_vec
        # print("k", k.shape)
        serie1 = self.serie1_vect(k, ttm_alone, N)
        # print(serie1.shape)
        serie2 = self.serie2_vect(k, ttm_alone, N)
        # print(serie2.shape)
        constant_term = np.exp(k - self.zeta * ttm_vec) - 1
        # print("const", constant_term.shape)
        # # factors
        factor_serie = np.exp(self.gamma * ttm_vec)
        factor = K * np.exp(-r * ttm_vec)
        call_price = factor * (constant_term + factor_serie * (serie1 + serie2))
        call_price = call_price[0, 0, 0, 0]
        if single_strike and single_ttm:
            call_price = call_price[0, 0]
            return call_price
        elif single_strike and not single_ttm:
            call_price = call_price[:, 0]
            return call_price
        elif not single_strike and single_ttm:
            call_price = call_price[0, :]
            return call_price
        elif not single_strike and not single_ttm:
            return call_price
        else:
            raise ValueError("Error")
