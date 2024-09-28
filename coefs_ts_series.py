"""Importing modules"""

import itertools as it
import numpy as np
import numpy.typing as npt

from scipy.special import gamma, factorial, poch


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

    def a1(
        self, n_vec: tuple, ttm: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """a1 coef in the paper

        Args:
            n_vec (tuple): n_i
            ttm (float): maturity

        Returns:
            float: a1 coef
        """
        n1, n2, n3, n4 = n_vec
        # taylor like term
        taylor = (-1) ** (n1 + n2 + n3) / (
            factorial(n1) * factorial(n2) * factorial(n3) * factorial(n4)
        )
        # pochammer symbols
        pochhamer_symb = (
            poch(-self.beta_m * n3, n1)
            * poch(n1 - self.beta_p * n2 - self.beta_m * n3, n4)
            / gamma(1 + self.beta_p * n2)
        )
        # gamma term
        if n2 == 0 and n3 == 0:
            gamma_term = ((-1) ** (1 + n1 + n4) / factorial(n1 + n4)) * (
                self.beta_p / (self.beta_p + self.beta_m)
            )
        else:
            gamma_term = gamma(-n1 + self.beta_p * n2 + self.beta_m * n3 - n4) / (
                gamma(-self.beta_p * n2)
            )
        # time term
        at_term = (self.ap * ttm) ** n2 * (self.am * ttm) ** n3

        return taylor * pochhamer_symb * gamma_term * at_term

    def a2(
        self, n_vec: tuple, ttm: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """a2 coef in the paper

        Args:
            n_vec (tuple): n_i
            ttm (float): maturity

        Returns:
            float: a1 coef
        """
        n1, n2, n3, n4 = n_vec
        # taylor like term
        taylor = (-1) ** (n1 + n2 + n3 + n4) / (
            factorial(n1) * factorial(n2) * factorial(n3) * factorial(n4)
        )
        # pochammer symbols
        pochhamer_symb = poch(1 + self.beta_p * n2, n1) / (-1 - n1 - n4)
        # gamma term
        if n2 == 0 and n3 == 0:
            gamma_term = 0
        else:
            gamma_term = gamma(-1 - n1 - self.beta_p * n2 - self.beta_m * n3) / (
                gamma(-self.beta_p * n2) * gamma(-self.beta_m * n3)
            )
        # time term
        at_term = (self.ap * ttm) ** n2 * (self.am * ttm) ** n3

        return taylor * pochhamer_symb * gamma_term * at_term

    def a3(
        self, n_vec: tuple, ttm: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """a3 coef in the paper

        Args:
            n_vec (tuple): n_i
            ttm (float): maturity

        Returns:
            float: a1 coef
        """
        n1, n2, n3 = n_vec
        # taylor like term
        taylor = -((-1) ** (n1 + n2)) / (factorial(n1) * factorial(n2))
        # pochammer symbols
        pochhamer_symb = 1 / gamma(1 - self.beta_p * n1 + n3)
        # gamma term
        if n2 == 0 and n3 == 0 and n1 == 0:
            gamma_term = self.beta_m / (self.beta_p + self.beta_m)
        else:
            gamma_term = gamma(-self.beta_p * n1 - self.beta_m * n2 + n3) / (
                gamma(-self.beta_m * n2)
            )
        # time term
        at_term = (self.ap * ttm) ** n1 * (self.am * ttm) ** n2
        return taylor * pochhamer_symb * gamma_term * at_term

    def get_nvecs(self, N: int, rep: int):
        itN = list(it.product(range(N), repeat=rep))
        return itN

    def price(self, S0: float, K: float, r: float, q: float, ttm: float, N: int = 15):
        # mettre k en array[None,:] et ttm en array [:,None]
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        # self.display_params(k)
        print(N)
        # series
        serie1 = self.serie1(k, ttm, N)
        serie2 = self.serie2(k, ttm, N)
        constant_term = np.exp(k - self.zeta * ttm) - 1
        # factors
        factor_serie = np.exp(self.gamma * ttm)
        factor = K * np.exp(-r * ttm)
        call_price = factor * (constant_term + factor_serie * (serie1 + serie2))
        return call_price

    def serie1(self, k: float, ttm: float, N: int):
        nvecs = self.get_nvecs(N, 4)
        serie = 0
        for nvec in nvecs:
            n1, n2, n3, n4 = nvec
            term1 = (
                self.a1(nvec, ttm)
                * self.ulambda**n1
                * (-k) ** (n1 - self.beta_p * n2 - self.beta_m * n3 + n4)
            )
            term2 = (
                self.a2(nvec, ttm)
                * self.ulambda ** (1 + n1 + self.beta_p * n2 + self.beta_m * n3)
                * (-k) ** (1 + n1 + n4)
            )
            exp_term = np.exp(k) * (self.lambda_p - 1) ** n4 - self.lambda_p**n4
            serie += (term1 + term2) * exp_term
        return serie

    def serie2(self, k: float, ttm: float, N: int):
        nvecs = self.get_nvecs(N, 3)
        serie = 0
        for nvec in nvecs:
            n1, n2, n3 = nvec
            term1 = self.a3(nvec, ttm) * self.ulambda ** (
                self.beta_p * n1 + self.beta_m * n2 - n3
            )
            exp_term = np.exp(k) * (self.lambda_p - 1) ** n3 - self.lambda_p**n3
            serie += (term1) * exp_term
        return serie
