import warnings
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial, poch
import scienceplots

plt.style.use("science")

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TSDensity:
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
        self.ulambda = lambda_m + lambda_p
        self.gamma = self.gamma_()
        self.std = self.get_std()
        self.mean = self.get_mean()

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

    def density_signed(
        self,
        points: float | npt.NDArray[np.float64],
        n: int = 20,
        positive: bool = True,
    ):
        if positive:
            lambda_p = self.lambda_p
            beta_p = self.beta_p
            beta_m = self.beta_m
            ap = self.ap
            am = self.am
        else:
            lambda_p = self.lambda_m
            beta_p = self.beta_m
            beta_m = self.beta_p
            ap = self.am
            am = self.ap

        if isinstance(points, float):
            x_vec = np.array([points])[None, None, None, :]
        else:
            x_vec = np.array(points)[None, None, None, :]
        n1 = np.arange(n)[:, None, None, None]
        n2 = np.arange(n)[None, :, None, None]
        n3 = np.arange(n)[None, None, :, None]
        prefact = np.exp(self.gamma - lambda_p * x_vec)
        taylor = (
            (-1) ** (n1 + n2 + n3)
            / (factorial(n1) * factorial(n2) * factorial(n3))
            * ap**n2
            * am**n3
        )

        c1 = poch(-beta_m * n3, n1) / gamma(1 + beta_p * n2)
        irr1 = gamma(1 - n1 + beta_p * n2 + beta_m * n3) / (gamma(-beta_p * n2))
        irr1[1:, 0, 0, :] = ((-1) ** n1 / factorial(n1 - 1))[1:, 0, 0, :]
        c1 = (
            c1
            * irr1
            * self.ulambda**n1
            * x_vec ** (-1 + n1 - beta_p * n2 - beta_m * n3)
        )
        c2 = poch(1 + beta_p * n2, n1)
        irr2 = gamma(-1 - n1 - beta_p * n2 - beta_m * n3) / (
            gamma(-beta_p * n2) * gamma(-beta_m * n3)
        )
        c2[:, 0, 0, :] = 0
        c2 = (
            c2
            * irr2
            * self.ulambda ** (1 + n1 + beta_p * n2 + beta_m * n3)
            * x_vec ** (n1)
        )
        c2[np.isnan(c2)] = 0
        dens = prefact * taylor * (c1 + c2)
        dens = dens.sum(axis=(0, 1, 2))
        return dens

    def density_Mellin(self, points: npt.NDArray[np.float64], n: int = 20):
        dens = np.zeros_like(points)
        dens[points > 0] = self.density_signed(points[points > 0], n=n, positive=True)
        dens[points < 0] = self.density_signed(-points[points < 0], n=n, positive=False)
        return dens

    def density_Fourier(
        self, points: npt.NDArray[np.float64], du: float = 1e-2, bounds: float = 3e3
    ):
        u = np.arange(-bounds, bounds, du)[None, :]
        integrand = np.exp(
            self.alpha_p
            * gamma(-self.beta_p)
            * ((self.lambda_p - 1j * u) ** self.beta_p - self.lambda_p**self.beta_p)
            + self.alpha_m
            * gamma(-self.beta_m)
            * ((self.lambda_m + 1j * u) ** self.beta_m - self.lambda_m**self.beta_m)
        ) * np.exp(-1j * u * points[:, None])
        return integrand.sum(axis=1) * du / (2 * np.pi)

    def get_mean(self):
        """return mean of TS

        Returns:
            float: mean
        """
        return gamma(1 - self.beta_p) * self.alpha_p / self.lambda_p ** (
            1 - self.beta_p
        ) - gamma(1 - self.beta_m) * self.alpha_m / self.lambda_m ** (1 - self.beta_m)

    def get_std(self):
        """return std

        Returns:
            float: std
        """
        var = gamma(2 - self.beta_p) * self.alpha_p / self.lambda_p ** (
            2 - self.beta_p
        ) + gamma(2 - self.beta_m) * self.alpha_m / self.lambda_m ** (2 - self.beta_m)
        std = var**0.5
        return std
