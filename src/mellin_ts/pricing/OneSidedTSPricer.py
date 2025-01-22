"""TBD"""

import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import factorial, gamma

# TODO: changer gamma upper par gamma - gamma lower
# pylint: enable=all
# pylint: disable=all
from src.gamma_func_cpp.lower_gamma_vect.gamma_incomp import (
    gamma_lower_incomplete_non_normalized,
)

# pylint: disable=all
from src.gamma_func_cpp.upper_gamma.gamma_module import (
    gamma_upper_incomplete as gamma_ui,
)
from src.gamma_func_cpp.upper_gamma_vect.gamma_module import (
    gamma_upper_incomplete as gamma_ui_vect,
)

# pylint: enable=all


def gamma_upper(a, z):
    return gamma(a) - gamma_lower_incomplete_non_normalized(a, z)


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
        # gamma_incomplete_vect = np.array(
        #     gamma_ui(-self.beta * n_vec, N * [-k * self.lambd])
        # )
        # gamma_incomplete_1_vect = np.array(
        #     gamma_ui(-self.beta * n_vec, N * [-k * (self.lambd - 1)])
        # )
        gamma_incomplete_vect = gamma_upper(-self.beta * n_vec, N * [-k * self.lambd])
        gamma_incomplete_1_vect = gamma_upper(
            -self.beta * n_vec, N * [-k * (self.lambd - 1)]
        )
        diff_vect = (
            np.exp(k)
            * (self.lambd - 1) ** (n_vec * self.beta)
            * gamma_incomplete_1_vect
            - self.lambd ** (self.beta * n_vec) * gamma_incomplete_vect
        )
        call_price = coef_vect * diff_vect
        call_price[0] = 0
        return call_price.sum()
