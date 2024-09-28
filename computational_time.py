"""importing modules"""

import time
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scienceplots

from coefs_ts_series import TemperedStablePricer

plt.style.use(["science"])


def main():
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.35,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
    )
    ts_pricer = TemperedStablePricer(**ts_params)
    option_params = dict(S0=1, K=[], r=0.02, q=0.05, ttm=[])
    times = compute_time_per_option(ts_pricer, option_params)
    print(times)
    plot_time(times)


def compute_time_per_option(
    ts_pricer: TemperedStablePricer, option_params: dict, n_test: int = 75
) -> dict:
    time_per_option = {}
    for n in range(n_test):
        strikes = np.linspace(1.1, 1.5, n)
        ttm = np.linspace(1.1, 1.3, n)
        option_params["K"] = strikes
        option_params["ttm"] = ttm
        t0 = time.time()
        ts_pricer.price(**option_params, N=15)
        t = time.time()
        time_per_option[n] = t - t0
    return time_per_option


def plot_time(times: dict):
    option_number = np.array(list(times.keys())) ** 2
    time_per_option = np.array(
        [time_comp / (n + 1) ** 2 for n, time_comp in times.items()]
    )
    plt.figure(figsize=(8, 5))
    plt.scatter(option_number, time_per_option)
    plt.xlabel(r"$N$")
    plt.ylabel("Time per option")
    plt.yscale("log")
    plt.grid()
    plt.savefig("output/computational_time.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
