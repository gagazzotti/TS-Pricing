"""Importing modules"""

# import time
# import tqdm
# import mpmath
# import itertools as it
import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import factorial, gamma

# pylint: disable=all
from src.gamma_func_cpp.upper_gamma.gamma_module import (
    gamma_upper_incomplete as gamma_ui,
)
from src.gamma_func_cpp.upper_gamma_vect.gamma_module import (
    gamma_upper_incomplete as gamma_ui_vect,
)

# TODO: changer gamma upper par gamma - gamma lower

# pylint: enable=all


warnings.filterwarnings("ignore")


class OneSidedTemperedStablePricer:
    """
    One sided Tempered Stable pricer
    """

    def __init__(
        self,
        alpha_p: float,
        beta_p: float,
        lambda_p: float,
    ):
        self.alpha = alpha_p
        self.beta = beta_p
        self.lambd = lambda_p
        self.ap = self.a(alpha_p, beta_p)
        self.zeta = self.zeta_()
        self.gamma = self.gamma_()

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
        zeta_p = (
            self.alpha
            * gamma(-self.beta)
            * ((self.lambd - 1) ** self.beta - self.lambd**self.beta)
        )
        return -zeta_p

    def gamma_(self) -> float:
        """gamma constant in the paper

        Returns:
            float: gamma
        """
        return self.ap * self.lambd**self.beta

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
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        serie = self.serie(k, ttm, N)
        call_price = K * np.exp((self.gamma - r) * ttm) * serie
        return call_price

    def serie(
        self,
        k: float | npt.NDArray[np.float64],
        ttm: float | npt.NDArray[np.float64],
        N: int,
    ):
        n_vec = np.arange(0, N)

        coef_vect = (-self.ap * ttm) ** n_vec / (
            factorial(n_vec) * gamma(-n_vec * self.beta)
        )
        gamma_incomplete_vect = np.array(
            gamma_ui(-self.beta * n_vec, N * [-k * self.lambd])
        )
        gamma_incomplete_1_vect = np.array(
            gamma_ui(-self.beta * n_vec, N * [-k * (self.lambd - 1)])
        )
        diff_vect = (
            np.exp(k)
            * (self.lambd - 1) ** (n_vec * self.beta)
            * gamma_incomplete_1_vect
            - self.lambd ** (self.beta * n_vec) * gamma_incomplete_vect
        )
        call_price = (coef_vect * diff_vect).sum()
        return call_price

    def price_vect(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        ttm: float,
        N: int = 25,
    ):
        if isinstance(ttm, float):
            ttm_vec = np.array([ttm])[None, :, None]
        else:
            ttm_vec = np.array(ttm)[None, :, None]
        if isinstance(K, float):
            K_vec = np.array([K])[None, None, :]
        else:
            K_vec = np.array(K)[None, None, :]
        k = np.log(S0 / K_vec) + (r - q + self.zeta) * ttm_vec
        serie = self.serie_vect(k, ttm_vec, N)
        call_price = K * np.exp((self.gamma - r) * ttm_vec) * serie
        return call_price

    def serie_vect(
        self,
        k: float | npt.NDArray[np.float64],
        ttm: float | npt.NDArray[np.float64],
        N: int,
    ):
        n_vec = np.arange(0, N)[:, None, None]

        coef_vect = (-self.ap * ttm) ** n_vec / (
            factorial(n_vec) * gamma(-n_vec * self.beta)
        )

        gamma_incomplete_vect = np.array(
            gamma_ui_vect(-self.beta * n_vec, -k[0] * self.lambd)
        )
        gamma_incomplete_1_vect = np.array(
            gamma_ui_vect(-self.beta * n_vec, -k[0] * (self.lambd - 1))
        )
        diff_vect = (
            np.exp(k)
            * (self.lambd - 1) ** (n_vec * self.beta)
            * gamma_incomplete_1_vect
            - self.lambd ** (self.beta * n_vec) * gamma_incomplete_vect
        )
        call_price = (coef_vect * diff_vect).sum(axis=(0))
        return call_price
