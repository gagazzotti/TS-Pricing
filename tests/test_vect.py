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
    N = 10
    ttm = np.arange(0.5, 1.5, 1e-2)
    K_vec = np.arange(1.2, 1.5, 1e-1)
    option_params = dict(S0=1, K=K_vec, r=0.02, q=0.05, ttm=ttm)
    t0 = time()
    ts_pricer = TemperedStablePricer(**ts_params)
    price = ts_pricer.price(**option_params, N=N)
    print(price.shape)
    ##
    option_params = dict(S0=1, K=1.5, r=0.02, q=0.05, ttm=1.2)
    t0 = time()
    price = ts_pricer.price(**option_params, N=50)
    print(price)


if __name__ == "__main__":
    main()
