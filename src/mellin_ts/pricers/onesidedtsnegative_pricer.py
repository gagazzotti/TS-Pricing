"""OneSidedTS Pricer"""

import warnings

import numpy as np
import scipy.special as sc
from scipy.integrate import quad

from src.gamma_func_cpp.gamma_lower import (
    gamma_lower as gamma_lower_cpp,
)


def gamma_upper(a, z):
    """upper gamma incomplete"""
    return sc.gamma(a) - gamma_lower_cpp(a, z)


warnings.filterwarnings("ignore")


def chf_ts(u, alpha, beta, lam):
    """
    Fonction caractéristique du tempered stable.

    Parameters
    ----------
    u : array_like
    alpha, beta, lam : floats

    Returns
    -------
    complex array
    """
    u = np.asarray(u, dtype=np.complex128)
    return np.exp(alpha * sc.gamma(-beta) * ((lam + 1j * u) ** beta - lam**beta))


def cdf_from_chf(x, alpha, beta, lam, u_max=1000, eps=1e-6):
    """
    CDF via inversion de la fonction caractéristique.

    Parameters
    ----------
    x : float
    alpha, beta, lam : floats
    u_max : cutoff intégration
    eps : tolérance

    Returns
    -------
    float
    """

    def integrand(u):
        phi = chf_ts(u, alpha, beta, lam)
        return np.imag(np.exp(-1j * u * x) * phi / u)

    integral, _ = quad(integrand, eps, u_max, limit=200)

    return 0.5 - (1 / np.pi) * integral


class OneSidedTemperedStablePricerNegative:
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
            * ((self.lambd + 1) ** self.beta - self.lambd**self.beta)
        )
        return -zeta_p

    def gamma_(self) -> float:
        """gamma constant in the paper

        Returns:
            float: gamma
        """
        return self.ap * self.lambd**self.beta

    def price(
        self, S0: float, K: float, r: float, q: float, ttm: float, N: int = 25
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
        # print("k", k)
        if k < 0:
            raise NotImplementedError("Negative moneyness not implemented so far.")
        elif k == 0:
            raise NotImplementedError("Negative moneyness not implemented so far.")
        else:
            serie = self.serie(S0, K, r, q, k, ttm, N)
            return serie

    def serie(
        self, S0: float, K: float, r: float, q: float, k: float, ttm: float, N: int
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
        # k = 0
        n_vec = np.arange(1, N)

        coef_vect = (-self.ap * ttm) ** n_vec / (
            sc.factorial(n_vec) * sc.gamma(-n_vec * self.beta)
        )
        # ruff: noqa: E731
        c_cn = lambda k: (
            np.exp(self.gamma * ttm)
            * coef_vect
            * self.lambd ** (self.beta * n_vec)
            * gamma_upper(-self.beta * n_vec, k * self.lambd)
        )
        c_an = lambda k: (
            np.exp(self.gamma * ttm)
            * coef_vect
            * (self.lambd + 1) ** (n_vec * self.beta)
            * gamma_upper(-self.beta * n_vec, k * (self.lambd + 1))
        )

        # ruff: noqa: enable
        # call_price = K * np.exp(k - r * ttm) * (
        #     np.exp(-self.zeta * ttm) - c_an(k).sum()
        # ) - K * np.exp(-r * ttm) * (1 - c_cn(k).sum())

        call_price = (
            S0 * np.exp(-q * ttm)
            - K * np.exp(-r * ttm)
            - K * np.exp(-r * ttm) * (np.exp(k) * c_an(k).sum() - c_cn(k).sum())
        )
        # print(
        #     "Serie sum",
        #     K * np.exp(-r * ttm) * (np.exp(k) * c_an(k).sum() - c_cn(k).sum()),
        # )

        # call_price = (
        #     S0 * np.exp(-q * ttm)
        #     - K * np.exp(-r * ttm)
        #     + K * np.exp(-r * ttm) * (c_cn(k).sum() - c_an(k).sum())
        # )
        # call_price[0] = 0
        # x = 1
        # print("CDF", 1 - cdf_from_chf(-x, self.alpha, self.beta, self.lambd))
        # print("CDF serie", 1 - c_cn(-x).sum())

        return call_price.sum()
