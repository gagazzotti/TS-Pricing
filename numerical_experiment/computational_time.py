"""importing modules"""

import time

# import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.ticker import FuncFormatter
import scienceplots
import tqdm

from src.mellin_ts.pricers.onesidedts_pricer import OneSidedTemperedStablePricer
from src.mellin_ts.pricers.ts_pricer import TemperedStablePricer

plt.style.use(["science"])


def main():
    """Save the graph of comp. time"""
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.35,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
    )
    ts_p_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        #     alpha_m=0.35,
        #     beta_m=0.5 - np.pi / 100,
        #     lambda_m=0.4,
    )
    ts_p_pricer = OneSidedTemperedStablePricer(**ts_p_params)
    ts_pricer = TemperedStablePricer(**ts_params)
    option_params = dict(S0=1, K=[], r=0.02, q=0.05, ttm=[])
    times = compute_time_per_option(ts_pricer, ts_p_pricer, option_params)
    print(times)
    plot_time(times)


def compute_time_per_option(
    ts_pricer: TemperedStablePricer,
    ts_p_pricer: OneSidedTemperedStablePricer,
    option_params: dict,
    n_test: int = 40,
) -> dict:
    """Return the time fo the price of n options

    Args:
        ts_pricer (OneSidedTemperedStablePricer): pricer
        option_params (dict): option paramaters
        n_test (int, optional): up to n_test. Defaults to 75.

    Returns:
        dict: {n**2:comp. time}
    """
    time_per_option = {"ts": {}, "ts_p": {}}
    for n in tqdm.tqdm(range(1, n_test)):
        # print(n)
        strikes = np.linspace(1.1, 1.5, n)
        ttm = np.linspace(1.1, 1.3, n)
        option_params["K"] = strikes
        option_params["ttm"] = ttm
        t0 = time.time()
        ts_pricer.price(**option_params, N=10)
        t = time.time()
        time_per_option["ts"][n] = t - t0
        t0 = time.time()
        ts_p_pricer.price_vect(**option_params, N=10)
        t = time.time()
        time_per_option["ts_p"][n] = t - t0
    return time_per_option


def plot_time(times: dict):
    """Save the plot

    Args:
        times (dict): {n:comp. time}
    """
    option_number = np.array(list(times["ts"].keys())) ** 2
    time_per_option_ts = np.array(
        [time_comp / (n + 1) ** 2 for n, time_comp in times["ts"].items()]
    )
    time_per_option_ts_p = np.array(
        [time_comp / (n + 1) ** 2 for n, time_comp in times["ts_p"].items()]
    )
    plt.figure(figsize=(8, 5))
    plt.scatter(
        option_number,
        time_per_option_ts,
        label="Double-sided TS",
        marker="x",
        color="green",
    )
    plt.scatter(
        option_number,
        time_per_option_ts_p,
        label="One-sided TS",
        marker="x",
        color="orange",
    )

    plt.xlabel("Number of options")
    plt.ylabel("Time per option")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.savefig(
        "numerical_experiment/output/computational_time_one_sided.png", dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
