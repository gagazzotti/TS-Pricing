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

    def price_(self, S0: float, K: float, r: float, q: float, ttm: float, N: int = 25):
        # mettre k en array[None,:] et ttm en array [:,None]
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        # t0 = time.time()
        serie1 = self.serie1(k, ttm, N)
        # print("Serie 1 non vect:", time.time() - t0)
        # t0 = time.time()
        serie1_vect = self.serie1_vect(k, ttm, N)
        # print("Serie 1 vect:", time.time() - t0)
        # print("Serie 1", serie1 - serie1_vect)
        # t0 = time.time()
        serie2 = self.serie2(k, ttm, N)
        # print("Time non vect", time.time() - t0)
        # t0 = time.time()
        serie2_vect = self.serie2_vect(k, ttm, N)
        # print("Time vect", time.time() - t0)
        # print(np.abs(serie2 - serie2_vect))
        constant_term = np.exp(k - self.zeta * ttm) - 1
        # factors
        factor_serie = np.exp(self.gamma * ttm)
        factor = K * np.exp(-r * ttm)
        call_price = factor * (constant_term + factor_serie * (serie1 + serie2))
        return call_price

    def serie1(self, k: float, ttm: float, N: int):
        nvecs = self.get_nvecs(N, 4)
        serie = 0
        a1_vect = self.a1_vect(N, ttm)
        a2_vect = self.a2_vect(N, ttm)

        for nvec in nvecs:
            n1, n2, n3, n4 = nvec
            term1 = (
                self.a1(nvec, ttm)
                * self.ulambda**n1
                * (-k) ** (n1 - self.beta_p * n2 - self.beta_m * n3 + n4)
            )
            # print(np.abs(self.a1(nvec, ttm) - a1_vect[n1, n2, n3, n4]) > 1e-10)
            term2 = (
                self.a2(nvec, ttm)
                * self.ulambda ** (1 + n1 + self.beta_p * n2 + self.beta_m * n3)
                * (-k) ** (1 + n1 + n4)
            )
            # print(np.abs(self.a2(nvec, ttm) - a2_vect[n1, n2, n3, n4]) > 1e-10)

            exp_term = np.exp(k) * (self.lambda_p - 1) ** n4 - self.lambda_p**n4
            serie += (term1 + term2) * exp_term
        return serie

    def serie2_vect(self, k: float, ttm: float, N: int):
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

    def serie1_vect(self, k: float, ttm: float, N: int):
        n1 = np.arange(N)[:, None, None, None]
        n2 = np.arange(N)[None, :, None, None]
        n3 = np.arange(N)[None, None, :, None]
        n4 = np.arange(N)[None, None, None, :]

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
        serie = ((term1 + term2) * exp_term).sum()
        return serie

    def a1_vect(self, N: int, ttm):
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
        gamma_term = gamma(-n1 + self.beta_p * n2 + self.beta_m * n3 - n4) / (
            gamma(-self.beta_p * n2)
        )
        gamma_term[:, 0, 0, :] = (
            ((-1) ** (1 + n1 + n4) / factorial(n1 + n4))
            * (self.beta_p / (self.beta_p + self.beta_m))
        )[:, 0, 0, :]
        a1 = pochhamer_symb * at_term * gamma_term * taylor
        return a1

    def a2_vect(self, N: int, ttm):
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
        # print(k)
        serie1_vect = self.serie1_vect(k, ttm, N)
        serie2_vect = self.serie2_vect(k, ttm, N)
        # print(serie1_vect, serie2_vect)
        constant_term = np.exp(k - self.zeta * ttm) - 1
        # factors
        factor_serie = np.exp(self.gamma * ttm)
        factor = K * np.exp(-r * ttm)
        call_price = factor * (
            constant_term + factor_serie * (serie1_vect + serie2_vect)
        )
        return call_price
