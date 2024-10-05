"""importing modules"""

from time import time
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import FuncFormatter

# import time

from mellin_ts.TemperedStablePricers import OneSidedTemperedStablePricer

plt.style.use(["science"])


def main():
    """Testing the convergence of the pricer"""
    ts_params = dict(
        alpha_p=0.4,
        beta_p=0.4 + np.exp(1) / 10,
        lambda_p=1.4,
        # alpha_m=0.05,
        # beta_m=0.5 - np.pi / 100,
        # lambda_m=0.4,
    )
    strike = 1.8
    ttm = 1
    option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
    ts_pricer = OneSidedTemperedStablePricer(**ts_params)
    N = 50
    print("N:", N)
    # t0 = time()
    # price = ts_pricer.price(**option_params, N=N)
    # print("Time non vect", time() - t0)
    t0 = time()
    price = ts_pricer.price(**option_params, N=N)
    print("Time", time() - t0)

    t0 = time()
    price_vect = ts_pricer.price_vect(**option_params, N=N)
    print("Time vect", time() - t0)
    print("Error:", price_vect - price)
    # print("Diff price:", price_vect - price)


if __name__ == "__main__":
    main()
