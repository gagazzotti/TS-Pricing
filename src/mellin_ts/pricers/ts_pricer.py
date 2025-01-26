"""TBD"""

import warnings

import numpy as np
import numpy.typing as npt
import scipy.special as sc

from src.gamma_func_cpp.gamma_lower import (
    gamma_lower as gamma_lower_cpp,
)

np.set_printoptions(precision=16)


warnings.filterwarnings("ignore", category=RuntimeWarning)

# TODO: faire un fichier de test gamma lower, faire un fichier d'import python de


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
        self.params = {"alpha_p": alpha_p, "beta_p": beta_p, "lambda_p": lambda_p,
                       "alpha_m": alpha_m, "beta_m": beta_m, "lambda_m": lambda_m}
        # transformed parameters
        # self.ap = -alpha_p * sc.gamma(-beta_p)
        # self.am = -alpha_m * sc.gamma(-beta_m)
        self.ulambda = lambda_m + lambda_p
        self.gamma = self.get_gamma()
        # convexity adjustment
        self.zeta = self.get_zeta()
        # term for computations (storing)
        self.bpn2 = None
        self.bmn3 = None
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

    def c1(self, k, n1, bpn2, bmn3, factor_otm):
        """TBD"""
        pochhamer_symb = sc.poch(-bmn3, n1) / sc.gamma(
            1 + bpn2
        )
        ulambda_term = self.ulambda ** (n1)
        linear_n = - n1 + bpn2 + bmn3
        gamma_term = sc.gamma(1 + linear_n) / (
            sc.gamma(-bpn2)
        )

        lower_gamma_an = np.exp(k) * (self.lambda_p-factor_otm) ** (linear_n) * gamma_lower_cpp(
            -linear_n,
            np.sign(k)*(self.lambda_p-factor_otm) * k
        )
        lower_gamma_cn = (self.lambda_p)**(linear_n) * \
            gamma_lower_cpp(-linear_n, np.sign(k)*(self.lambda_p) * k)
        low_gamma_term = gamma_term*(lower_gamma_an-lower_gamma_cn)

        # multiplication par e^{k}-1
        low_gamma_term[0, 0, 0] = (
            self.beta_p/(self.beta_m + self.beta_p) * (np.exp(k) - 1))

        # gamma_term in >1,0,0
        low_gamma_term[1:, 0, 0] = 0
        full_a1 = pochhamer_symb * ulambda_term * low_gamma_term
        return full_a1

    def c2(self, k, n1, bpn2, bmn3, factor_otm):
        """TBD"""
        pochhamer_symb = sc.poch(1 + bpn2, n1)
        linear_n = -1 - n1 - bpn2 - bmn3
        gamma_term = sc.gamma(linear_n) / \
            (sc.gamma(-bpn2) * sc.gamma(-bmn3))
        gamma_term[:, 0, 0] = 0
        # time term
        a2 = pochhamer_symb * gamma_term
        ulambda_term = self.ulambda ** (-linear_n)
        lower_gamma_an = np.exp(k) * (self.lambda_p-factor_otm) ** (-1 - n1) * \
            (gamma_lower_cpp(1 + n1, np.sign(k)*(self.lambda_p-factor_otm) * k))
        lower_gamma_cn = (self.lambda_p) ** (-1 - n1) * \
            (gamma_lower_cpp(1 + n1, np.sign(k)*self.lambda_p * k))
        function_term = lower_gamma_an-lower_gamma_cn
        full_a2 = a2 * function_term * ulambda_term
        return full_a2

    def c3(self, k: float, n1: npt.NDArray, bpn2, bmn3, factor_otm):
        """TBD"""
        gamma_term = sc.gamma(-bpn2 - bmn3 + n1) / (
            sc.gamma(1 - bpn2 + n1) * sc.gamma(-bmn3)
        )
        gamma_term[0, 0, 0] = self.beta_m / (self.beta_m + self.beta_p)
        ulambda_term = self.ulambda ** (bpn2 + bmn3 - n1)
        exp_term = np.exp(k) * (self.lambda_p -
                                factor_otm) ** n1 - self.lambda_p**n1
        serie = (gamma_term * exp_term * ulambda_term)
        return serie

    def serie(
        self, k: float, ttm: float, n1: npt.NDArray, n2: npt.NDArray, n3: npt.NDArray
    ):
        """TBD"""
        is_otm = k < 0
        if not is_otm:
            self.reverse_params()
        # current term that will be repeatedly called
        bpn2 = self.beta_p*n2
        bmn3 = self.beta_m*n3
        at = (self.ap * ttm) ** n2 * (self.am * ttm) ** n3
        # taylor term
        fact_n2n3 = ((-1) ** (n2 + n3)) / (sc.factorial(n2) * sc.factorial(n3))
        fact_n1 = (-1)**n1 / sc.factorial(n1)
        factor_otm = 1 if is_otm else -1
        term1 = self.c1(k, n1, bpn2, bmn3, factor_otm)
        term2 = self.c2(k, n1, bpn2, bmn3, factor_otm)
        term3 = self.c3(k, n1, bpn2, bmn3, factor_otm)
        serie = (at*fact_n2n3*(fact_n1*(term1 + term2) + term3)
                 ).sum(axis=(0, 1, 2))
        if not is_otm:
            self.reverse_params()

        return -serie

    def price(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        ttm: float,
        N: int = 5,
    ):
        """TBD"""
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        n1, n2, n3 = self.get_n_vectors(N)
        serie = self.serie(k, ttm, n1, n2, n3)
        call_price = K * np.exp(-r * ttm) * \
            (np.exp(k - self.zeta * ttm) - 1 + np.exp(self.gamma * ttm) * serie)
        if k > 0:
            call_price_itm = S0*np.exp(-q*ttm)-K*np.exp(-r*ttm)-call_price
            return call_price_itm
        elif k < 0:
            return call_price
        else:
            raise NotImplementedError(
                "Moneyness equal to 0 not implemented yet.")

    def reverse_params(self):
        """reverse params"""
        self.params = {"alpha_p": self.alpha_m,
                       "beta_p": self.beta_m,
                       "lambda_p": self.lambda_m,
                       "alpha_m": self.alpha_p,
                       "beta_m": self.beta_p,
                       "lambda_m": self.lambda_p}

    def get_n_vectors(self, N: int):
        """get n vectors"""
        n1 = np.arange(N)[:, None, None]
        n2 = np.arange(N)[None, :, None]
        n3 = np.arange(N)[None, None, :]
        return n1, n2, n3

    @ property
    def alpha_p(self):
        """return the alpha_p parameter"""
        return self.params["alpha_p"]

    @ property
    def beta_p(self):
        """return the beta_p parameter"""
        return self.params["beta_p"]

    @ property
    def lambda_p(self):
        """return the lambda_p parameter"""
        return self.params["lambda_p"]

    @ property
    def alpha_m(self):
        """return the alpha_m parameter"""
        return self.params["alpha_m"]

    @ property
    def beta_m(self):
        """return the beta_m parameter"""
        return self.params["beta_m"]

    @ property
    def lambda_m(self):
        """return the lambda_m parameter"""
        return self.params["lambda_m"]

    @ property
    def ap(self):
        """ap"""
        return -self.alpha_p * sc.gamma(-self.beta_p)

    @ property
    def am(self):
        """am"""
        return -self.alpha_m * sc.gamma(-self.beta_m)
