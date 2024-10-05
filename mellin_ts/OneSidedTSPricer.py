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

    def price_old(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        ttm: float,
        N: int = 25,
        # time_verbose=True,
    ):
        # mettre k en array[None,:] et ttm en array [:,None]
        # t0 = time.time()
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        # print(f"k:{k}")
        # self.display_params(k)
        serie = self.serie(k, ttm, N)
        call_price = K * np.exp((self.gamma - r) * ttm) * serie
        # if time_verbose:
        #     print(f"Time: {time.time()-t0}")
        return call_price

    def serie(
        self,
        k: float | npt.NDArray[np.float64],
        ttm: float | npt.NDArray[np.float64],
        N: int,
    ):

        gamma_inc_np = np.frompyfunc(mpmath.gammainc, 2, 1)
        serie = 0
        for n in range(N):
            coef = (-self.ap * ttm) ** n / (factorial(n) * gamma(-n * self.beta))
            # print(mpmath.gammainc(-self.beta * n, -k * self.lambd))
            gam_lam = np.array(gamma_inc_np(-self.beta * n, -k * self.lambd)).astype(
                float
            )
            # print(gamma(-n * self.beta) * gamma_inc_np(-self.beta * n, -k * self.lambd))
            gam_lam_1 = np.array(
                gamma_inc_np(-self.beta * n, -k * (self.lambd - 1))
            ).astype(float)
            diff = (
                np.exp(k) * (self.lambd - 1) ** (n * self.beta) * gam_lam_1
                - self.lambd ** (self.beta * n) * gam_lam
            )
            serie += coef * diff
            # print(
            #     gamma_inc_np(-self.beta * n, -k * self.lambd)
            #     - gamma_upper_incomplete([-self.beta * n], [-k * self.lambd])[0]
            # )

        ##
        ##

        return serie

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
        # mettre k en array[None,:] et ttm en array [:,None]
        # t0 = time.time()
        k = np.log(S0 / K) + (r - q + self.zeta) * ttm
        # print(f"k:{k}")
        # self.display_params(k)
        serie = self.serie_vect(k, ttm, N)
        call_price = K * np.exp((self.gamma - r) * ttm) * serie
        # if time_verbose:
        #     print(f"Time: {time.time()-t0}")
        return call_price

    def serie_vect(
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
            gamma_upper_incomplete(-self.beta * n_vec, N * [-k * self.lambd])
        )
        gamma_incomplete_1_vect = np.array(
            gamma_upper_incomplete(-self.beta * n_vec, N * [-k * (self.lambd - 1)])
        )
        diff_vect = (
            np.exp(k)
            * (self.lambd - 1) ** (n_vec * self.beta)
            * gamma_incomplete_1_vect
            - self.lambd ** (self.beta * n_vec) * gamma_incomplete_vect
        )
        call_price = (coef_vect * diff_vect).sum()
        return call_price
