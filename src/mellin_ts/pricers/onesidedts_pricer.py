"""TBD"""

import warnings

import numpy as np
import scipy.special as sc

from src.gamma_func_cpp.gamma_lower import (
    gamma_lower as gamma_lower_cpp,
)


def gamma_upper(a, z):
    """TBD"""
    return sc.gamma(a) - gamma_lower_cpp(a, z)


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
        return -alpha * sc.gamma(-beta)

    def zeta_(self):
        """convexity adjustment

        Returns:
            zeta: convex. adj
        """
        zeta_p = (
            self.alpha
            * sc.gamma(-self.beta)
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
        N: int = 25
    ) -> float:
        """
        Main function to price a call.

        Args:
            S0 (float): S0
            K (float): strike
            r (float): free-risk rate
            q (float): dividend rate
            ttm (float): timt to maturity
            N (int, optional): order of summation. Defaults to 25.

        Returns:
            float: call price
        """
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        if k > 0:
            raise NotImplementedError(
                "Negative moneyness not implemented so far.")
        elif k == 0:
            raise NotImplementedError(
                "Negative moneyness not implemented so far.")
        else:
            serie = self.serie(k, ttm, N)
            call_price = K * np.exp((self.gamma - r) * ttm) * serie
            return call_price

    def serie(
        self,
        k: float,
        ttm: float,
        N: int
    ) -> float:
        """
        Compute the serie exposed in the article.

        Args:
            k (float): moneyness
            ttm (float): _description_
            N (int): _description_

        Returns:
            float: _description_
        """
        n_vec = np.arange(0, N)

        coef_vect = (-self.ap * ttm) ** n_vec / (
            sc.factorial(n_vec) * sc.gamma(-n_vec * self.beta)
        )
        gamma_incomplete_vect = gamma_upper(-self.beta *
                                            n_vec, -k * self.lambd)
        gamma_incomplete_1_vect = gamma_upper(
            -self.beta * n_vec, -k * (self.lambd - 1)
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
