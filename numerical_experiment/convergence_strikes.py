"""TBD"""

import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scienceplots
import tqdm
from fypy.model.levy.TemperedStable import TemperedStable
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward
from matplotlib.ticker import FuncFormatter

from src.mellin_ts.pricers.onesidedts_pricer import OneSidedTemperedStablePricer
from src.mellin_ts.pricers.ts_pricer import TemperedStablePricer

plt.style.use(["science"])


def compute_moneyness(option_params: dict, zeta: float):
    """TBD"""
    drift = (option_params["r"]-option_params["q"]+zeta)*option_params["ttm"]
    return np.log(option_params["S0"]/option_params["strikes"]) + drift


def main():
    """Testing the convergence of the pricer"""
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.35,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
    )
    n_start, n_end = 1, 101
    range_n = np.arange(n_start, n_end, 5) - 1
    range_n[0] = 1
    range_n = list(range_n)
    ttm = 0.2
    option_params = dict(S0=1, r=0.02, q=0.05, ttm=ttm,
                         strikes=np.arange(0.5, 2.2, 0.1))
    ts_pricer = TemperedStablePricer(**ts_params)
    zeta = ts_pricer.zeta
    moneyness = compute_moneyness(option_params, zeta)
    prices = get_proj_prices(option_params, ts_params)

    plt.plot(option_params["strikes"], prices)
    plt.plot(option_params["strikes"], (option_params["S0"]-option_params["strikes"])
             * (option_params["S0"]-option_params["strikes"] >= 0))
    plt.grid()
    plt.xlabel(r"$K$")
    plt.show()


def decreasing_error(
    prices: npt.NDArray[np.float64], range_n: list[int]
):
    """Convergence to PROJ price, plotting error

    Args:
        ts_pricer (TemperedStablePricer): pricer
        option_params (dict): standard params
        n_start (int, optional): n to begin with. Defaults to 10.
        n_end (int, optional): n to end with. Defaults to 16.
    """
    proj_price_ref = 0.22968572289949263
    proj_price_ref_p = 0.18769860488552348
    rel_err_ts = np.abs(prices["ts"] - proj_price_ref) / proj_price_ref
    rel_err_p = np.abs(prices["ts_p"] - proj_price_ref_p) / proj_price_ref_p

    # Create the figure and the first Y-axis (for the relative error)
    _, ax1 = plt.subplots(figsize=(8, 5))
    # relative_error = 3 * np.array(range_n)
    plt.scatter(
        range_n,
        rel_err_ts,
        color="green",
        label="Series price error (Double-sided)",
        marker="x",
    )
    plt.scatter(
        range_n,
        rel_err_p,
        color="orange",
        label="Series price error (One-sided)",
        marker="x",
    )

    # Plot relative error on the primary y-axis
    step = 5
    xticks = np.arange(min(range_n), max(range_n) + 1, step) - 1
    xticks[0] = 1
    plt.xticks(xticks)
    ax1.grid()
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$N$")
    ax1.set_xlim(min(range_n) - 0.2, max(range_n) + 0.2)
    ax1.set_ylabel("Relative Error")
    ax1.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x)}")
    )  # For the X-axis
    ax1.set_xticks(xticks)  # X-ticks every 5 units
    ax1.legend(loc="upper left")

    # Create the second Y-axis (for the time data)

    # Save the figure
    plt.savefig(
        "numerical_experiment/output/test.png", dpi=200)
    plt.close()


def convergence(prices: npt.NDArray[np.float64], range_n: list[int]):
    """Convergence to PROJ price, plotting error

    Args:
        ts_pricer (TemperedStablePricer): pricer
        option_params (dict): standard params
        n_start (int, optional): n to begin with. Defaults to 10.
        n_end (int, optional): n to end with. Defaults to 16.
    """
    proj_price_ref = 0.22967873723167465
    proj_price_ref_p = 0.18769876719600298

    # proj_price_ref = 0.22968572289948497
    # proj_price_ref_p = 0.18769860488552348

    plt.figure(figsize=(8, 5))
    plt.scatter(
        range_n,
        prices["ts"],
        marker="x",
        color="green",
        label="Series Price (Double-sided)",
    )
    plt.scatter(
        range_n,
        prices["ts_p"],
        marker="x",
        color="orange",
        label="Series Price (One-sided)",
    )

    plt.hlines(
        proj_price_ref,
        min(range_n) - 1,
        max(range_n) + 1,
        color="green",
        label="Reference Price (Double-sided)",
    )
    plt.hlines(
        proj_price_ref_p,
        min(range_n) - 1,
        max(range_n) + 1,
        color="orange",
        label="Reference Price (One-sided)",
    )
    # plt.ylim(0.15, 0.3)
    plt.ylim(0, 0.44)
    step = 5
    xticks = np.arange(min(range_n), max(range_n) + 1, step) - 1
    xticks[0] = 1
    plt.xticks(xticks)
    plt.xlim(min(range_n) - 0.2, max(range_n) + 0.2)
    plt.xlabel(r"$N$")
    plt.ylabel("Call Price")
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x)}")
    )  # For the X-axis
    # plt.xticks(range_n)
    plt.savefig("numerical_experiment/output/convergence_test.png", dpi=200)
    plt.close()


def get_proj_prices(
    option_params: dict,
    ts_params: dict,
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
    # PROJ framewrok
    disc_curve = DiscountCurve_ConstRate(rate=option_params["r"])
    div_disc = DiscountCurve_ConstRate(rate=option_params["q"])
    fwd = EquityForward(
        S0=option_params["S0"], discount=disc_curve, divDiscount=div_disc)
    model = TemperedStable(
        forwardCurve=fwd, discountCurve=disc_curve, **ts_params)
    proj_pricer = ProjEuropeanPricer(
        model=model, N=2**22, order=3, alpha_override=500)
    prorj_prices = proj_pricer.price_strikes(
        option_params["ttm"], option_params["strikes"], [True]*len(option_params["strikes"]))
    return prorj_prices


if __name__ == "__main__":
    main()
