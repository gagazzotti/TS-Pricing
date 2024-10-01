"""importing modules"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scienceplots

from matplotlib.ticker import FuncFormatter

# import time

from TemperedStablePricers import OneSidedTemperedStablePricer

plt.style.use(["science"])


def main():
    """Testing the convergence of the pricer"""
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        # alpha_m=0.35,
        # beta_m=0.5 - np.pi / 100,
        # lambda_m=0.4,
    )
    n_start, n_end = 1, 20
    range_n = list(range(n_start, n_end))
    strike = 1.5
    ttm = 1.2
    option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
    ts_pricer = OneSidedTemperedStablePricer(**ts_params)
    prices = get_prices(ts_pricer, option_params, range_n)

    decreasing_error(prices, range_n)
    convergence(prices, range_n)


def decreasing_error(prices: npt.NDArray[np.float64], range_n: list[int]):
    """Convergence to PROJ price, plotting error

    Args:
        ts_pricer (TemperedStablePricer): pricer
        option_params (dict): standard params
        n_start (int, optional): n to begin with. Defaults to 10.
        n_end (int, optional): n to end with. Defaults to 16.
    """
    proj_price_ref = 0.1876986048855046

    plt.figure(figsize=(8, 5))
    plt.scatter(
        range_n,
        np.abs(prices - proj_price_ref) / proj_price_ref,
        color="black",
        label="Series Price",
    )
    plt.grid()
    plt.yscale("log")
    plt.xlabel(r"$N$")
    plt.xlim(min(range_n) - 0.2, max(range_n) + 0.2)
    plt.ylabel("Relative Error")
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x)}")
    )  # For the X-axis
    plt.xticks(range_n)
    plt.savefig("output/decreasing_error_one_sided.png", dpi=200)
    plt.close()


def convergence(prices: npt.NDArray[np.float64], range_n: list[int]):
    """Convergence to PROJ price, plotting error

    Args:
        ts_pricer (TemperedStablePricer): pricer
        option_params (dict): standard params
        n_start (int, optional): n to begin with. Defaults to 10.
        n_end (int, optional): n to end with. Defaults to 16.
    """
    proj_price_ref = 0.18769876719600298

    plt.figure(figsize=(8, 5))
    plt.scatter(range_n, prices, marker="x", color="black", label="Series Price")
    plt.hlines(
        proj_price_ref,
        min(range_n) - 1,
        max(range_n) + 1,
        color="green",
        label="Reference Price",
    )
    plt.ylim(0.15, 0.3)
    plt.xlim(min(range_n) - 0.2, max(range_n) + 0.2)
    plt.xlabel(r"$N$")
    plt.ylabel("Call Price")
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x)}")
    )  # For the X-axis
    plt.xticks(range_n)
    plt.savefig("output/convergence_one_sided.png", dpi=200)
    plt.close()


def get_prices(
    ts_pricer: OneSidedTemperedStablePricer, option_params: dict, range_n: list[int]
):
    """_summary_

    Args:
        ts_pricer (TemperedStablePricer): pricer
        option_params (dict): standard params
        n_start (int, optional): n to begin with. Defaults to 10.
        n_end (int, optional): n to end with. Defaults to 16.

    Returns:
        prices
    """
    prices = np.array([ts_pricer.price(**option_params, N=n) for n in range_n])
    return prices


if __name__ == "__main__":
    main()
