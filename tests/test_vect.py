"""importing modules"""

from time import time
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import FuncFormatter

# import time

from mellin_ts.TSPricer import TemperedStablePricer

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
    N = 20
    strike = 1.1
    ttm = np.arange(0.5, 1.5, 1e-2)
    t0 = time()
    prices = []
    for t in ttm:
        option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=t)
        ts_pricer = TemperedStablePricer(**ts_params)
        price = ts_pricer.price(**option_params, N=N)
        prices.append(price)
    print("Time", time() - t0)

    option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
    t0 = time()
    price_vect = ts_pricer.price_vect(**option_params, N=N)
    print(
        "Time",
        time() - t0,
        "Diff prices L1",
        np.abs(np.array(prices) - price_vect[0, 0, 0, 0]).sum(),
    )


if __name__ == "__main__":
    main()
