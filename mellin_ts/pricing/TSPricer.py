"""Importing modules"""

import warnings
from time import time

import numpy as np
import numpy.typing as npt
import scipy
import scipy.special
from scipy.special import factorial, gamma, poch

np.set_printoptions(precision=16)

# pylint: disable=all

from mellin_ts.pricing.lower_gamma_vect.gamma_incomp import (
    gamma_lower_incomplete_non_normalized,
)
from mellin_ts.pricing.upper_gamma.gamma_module import (
    gamma_upper_incomplete as gamma_ui,
)
from mellin_ts.pricing.upper_gamma_vect.gamma_module import (
    gamma_upper_incomplete as gamma_ui_vect,
)

# pylint: enable=all


warnings.filterwarnings("ignore", category=RuntimeWarning)

# TODO: enlever gamma lower scipy


def gamma_lower_scipy(a, z):
    return scipy.special.gamma(a) * (1 - scipy.special.gammaincc(a, z))


def gamma_lower_cpp(a, z):
    return gamma_lower_incomplete_non_normalized(a, z)


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
        n4 = np.arange(100)[
            None, None, None, :, None, None
        ]  # on peut mettre 80 partout pour essayer de voir coef par coef sur les 3
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
        n4 = np.arange(100)[None, None, None, :, None, None]
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

    def a2_vect_3_indexes(self, N: int, ttm, k):
        ttm_vec = np.array(ttm)[None, None, None, :, None]
        n1 = np.arange(N)[:, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None]
        taylor = (-1) ** (n1 + n2 + n3) / (
            factorial(n1) * factorial(n2) * factorial(n3)
        )
        pochhamer_symb = poch(1 + self.beta_p * n2, n1)
        gamma_term = gamma(-1 - n1 - self.beta_p * n2 - self.beta_m * n3) / (
            gamma(-self.beta_p * n2) * gamma(-self.beta_m * n3)
        )
        gamma_term[:, 0, 0, :] = 0
        # time term
        at_term = (self.ap * ttm_vec) ** n2 * (self.am * ttm_vec) ** n3
        a2 = taylor * pochhamer_symb * gamma_term * at_term
        ulambda_term = self.ulambda ** (1 + n1 + self.beta_p * n2 + self.beta_m * n3)
        k_vec = k[
            :, :, 0
        ]  # doit faire ca car A1 pas encore vectorisé, on pourra réduire les dimensions une fois A1 fait
        function_term = (
            np.exp(k_vec)
            * (self.lambda_p - 1) ** (-1 - n1)
            * (gamma_lower_scipy(1 + n1, -(self.lambda_p - 1) * k_vec))
        ) - (
            (self.lambda_p) ** (-1 - n1)
            * (gamma_lower_scipy(1 + n1, -(self.lambda_p) * k_vec))
        )
        return -a2 * function_term * ulambda_term

    def a1_vect_3_indexes(self, N: int, ttm, k):
        ttm_vec = np.array(ttm)[None, None, None, :, None]
        n1 = np.arange(N)[:, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None]
        taylor = (-1) ** (n1 + n2 + n3) / (
            factorial(n1) * factorial(n2) * factorial(n3)
        )
        pochhamer_symb = poch(-self.beta_m * n3, n1) / gamma(1 + self.beta_p * n2)

        at_term = (self.ap * ttm_vec) ** n2 * (self.am * ttm_vec) ** n3
        ulambda_term = self.ulambda ** (n1)
        k_vec = k[:, :, 0]  # doit faire ca car
        # A1 pas encore vectorisé, on pourra réduire les dimensions une fois A1 fait
        k_vec = np.ones_like(n1 + n2 + n3).astype(float) * k_vec.item()
        # piecwise
        low_gamma_term = np.zeros_like(n1 + n2 + n3).astype(float)
        # gamma_term in 0,0,0

        # other
        low_gamma_term = gamma(1 - n1 + self.beta_p * n2 + self.beta_m * n3) / (
            gamma(-self.beta_p * n2)
        )

        low_gamma_term = low_gamma_term * (
            (
                np.exp(k_vec)
                * (self.lambda_p - 1) ** (-n1 + self.beta_p * n2 + self.beta_m * n3)
                * np.array(
                    gamma_lower_cpp(
                        n1 - self.beta_p * n2 - self.beta_m * n3,
                        -(self.lambda_p - 1) * k_vec,
                    )
                )
            )
            - (
                (self.lambda_p) ** (-n1 + self.beta_p * n2 + self.beta_m * n3)
                * np.array(
                    gamma_lower_cpp(
                        n1 - self.beta_p * n2 - self.beta_m * n3,
                        -(self.lambda_p) * k_vec,
                    )
                )
            )
        ).astype(float)

        # multiplication par e^{k}-1
        low_gamma_term[0, 0, 0, :] = (
            self.beta_p
            / (self.beta_m + self.beta_p)
            * (np.exp(k_vec[0, 0, 0]) - 1).astype(float)
        )

        # gamma_term in >1,0,0
        low_gamma_term[1:, 0, 0, :] = 0

        a1 = taylor * pochhamer_symb * at_term

        return -a1 * ulambda_term * low_gamma_term

    def serie1_vect(self, k: float, ttm: float, N: int):
        ttm_vec = np.array(ttm)[None, None, None, :, None]
        factor_serie = 1  # np.exp(self.gamma * ttm)
        # faire la multiplication seuelement à la fin
        n1 = np.arange(N)[:, None, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None, None]
        n4 = np.arange(100)[None, None, None, :, None, None]

        #####################################################
        ################## 4 indexes SERIES #################
        #####################################################
        if False:
            t0 = time()
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

            # serie1_true = factor_serie[:, None] * (term1 * exp_term).sum(axis=(0, 1, 2, 3))
            # serie2_true = factor_serie[:, None] * (term2 * exp_term).sum(axis=(0, 1, 2, 3))
            serie = (term1 + term2) * exp_term
            serie = factor_serie[:, None] * serie.sum(axis=(0, 1, 2, 3))
            print("Time 4 indexes", time() - t0)

        #####################################################
        ################## SECOND SERIE #####################
        #####################################################

        # 3 indexes
        # t0 = time()
        term1_3_index = self.a1_vect_3_indexes(N, ttm, k)
        serie1 = (term1_3_index).sum(axis=(0, 1, 2))
        term2_3_index = self.a2_vect_3_indexes(N, ttm, k)
        serie2 = (term2_3_index).sum(axis=(0, 1, 2))
        serie = serie1 + serie2
        # print("Time 3 indexes", time() - t0)
        return serie1 + serie2

    def serie2_vect(self, k: float, ttm: float, N: int):
        k_vec = k[0]
        ttm_vec = np.array(ttm)[None, None, None, :, None]
        n1 = np.arange(N)[:, None, None, None, None]
        n2 = np.arange(N)[None, :, None, None, None]
        n3 = np.arange(N)[None, None, :, None, None]
        taylor_term = -((-1) ** (n2 + n3)) / (factorial(n2) * factorial(n3))
        gamma_term = np.zeros((N, N, N))
        gamma_term = gamma(-self.beta_p * n2 - self.beta_m * n3 + n1) / (
            gamma(1 - self.beta_p * n2 + n1) * gamma(-self.beta_m * n3)
        )
        gamma_term[0, 0, 0] = self.beta_m / (self.beta_m + self.beta_p)
        ulambda_term = self.ulambda ** (self.beta_p * n2 + self.beta_m * n3 - n1)
        exp_term = np.exp(k_vec) * (self.lambda_p - 1) ** n1 - self.lambda_p**n1
        at = (self.ap * ttm_vec) ** n2 * (self.am * ttm_vec) ** n3
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
        ##
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
