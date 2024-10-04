"""importing modules"""

import time
import numpy as np
import matplotlib.pyplot as plt

from MellinTemperedStable.TemperedStablePricers import TemperedStablePricer


def main():
    """
    Test of TS pricing
    """
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.35,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
    )
    strikes = np.arange(1.2, 20, 0.2)[None, :]
    times = np.arange(1, 2, 0.1)[:, None]
    option_params = dict(S0=1, K=strikes, r=0.02, q=0.05, ttm=times)
    ts_pricer = TemperedStablePricer(**ts_params)
    t0 = time.time()
    prices = ts_pricer.price(**option_params, N=25)
    t = time.time()
    print(prices.shape, f"Time: {t-t0}")
    for i in range(prices.shape[0]):
        plt.plot(strikes[0], prices[i], label=f"T={times[i]}")
    plt.legend()
    plt.grid()
    plt.xlabel("Strike")
    plt.ylabel("Call price")
    plt.show()


if __name__ == "__main__":
    main()
