"""importing modules"""

import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scienceplots
import tqdm
from matplotlib.ticker import FuncFormatter

from src.mellin_ts.pricing.OneSidedTSPricer import OneSidedTemperedStablePricer
from src.mellin_ts.pricing.TSPricer import TemperedStablePricer

plt.style.use(["science"])


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
    ts_p_params = dict(alpha_p=0.44, beta_p=0.1 + np.exp(1) / 10, lambda_p=1.4)
    n_start, n_end = 1, 81
    range_n = np.arange(n_start, n_end, 5) - 1
    range_n[0] = 1
    range_n = list(range_n)
    strike = 1.5
    ttm = 1.2
    option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
    ts_pricer = TemperedStablePricer(**ts_params)
    ts_p_pricer = OneSidedTemperedStablePricer(**ts_p_params)

    prices, times = get_prices(ts_pricer, ts_p_pricer, option_params, range_n)

    decreasing_error(prices, times, range_n)
    convergence(prices, range_n)


def decreasing_error(
    prices: npt.NDArray[np.float64], times: npt.NDArray[np.float64], range_n: list[int]
):
    """Convergence to PROJ price, plotting error

    Args:
        ts_pricer (TemperedStablePricer): pricer
        option_params (dict): standard params
        n_start (int, optional): n to begin with. Defaults to 10.
        n_end (int, optional): n to end with. Defaults to 16.
    """
    proj_price_ref = 0.22968572289948497
    proj_price_ref_p = 0.18769860488552348
    rel_err_ts = np.abs(prices["ts"] - proj_price_ref) / proj_price_ref
    rel_err_p = np.abs(prices["ts_p"] - proj_price_ref_p) / proj_price_ref_p

    # Create the figure and the first Y-axis (for the relative error)
    fig, ax1 = plt.subplots(figsize=(8, 5))
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
    ax2 = ax1.twinx()
    ax2.set_ylabel("Computation Time (s)")
    ax2.plot(
        range_n, times["ts"], label="Time TS (Double-sided)", color="green", alpha=0.5
    )
    ax2.plot(
        range_n, times["ts_p"], label="Time TS (One-sided)", color="orange", alpha=0.5
    )
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")

    # Save the figure
    plt.savefig("numerical_experiment/output/decreasing_error_with_time.png", dpi=200)
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
    plt.savefig("numerical_experiment/output/convergence.png", dpi=200)
    plt.close()


def get_prices(
    ts_pricer: TemperedStablePricer,
    ts_p_pricer: OneSidedTemperedStablePricer,
    option_params: dict,
    range_n: list[int],
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
    prices = {}
    times = {}
    price_ts = []
    time_ts = []
    for n in tqdm.tqdm(range_n):
        t0 = time.time()
        price_ts.append(ts_pricer.price(**option_params, N=n))
        time_ts.append(time.time() - t0)
    prices["ts"] = np.array(price_ts)
    times["ts"] = np.array(time_ts)
    price_ts_p = []
    time_ts_p = []
    for n in tqdm.tqdm(range_n):
        t0 = time.time()
        price_ts_p.append(ts_p_pricer.price(**option_params, N=n))
        time_ts_p.append(time.time() - t0)
    prices["ts_p"] = np.array(price_ts_p)
    times["ts_p"] = np.array(time_ts_p)
    return prices, times


if __name__ == "__main__":
    main()
