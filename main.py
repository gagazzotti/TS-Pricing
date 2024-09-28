import numpy as np

from coefs_ts_series import TemperedStablePricer
import time


def main():
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.35,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
    )
    strikes = np.arange(1.2, 1.8, 0.01)
    option_params = dict(S0=1, K=strikes, r=0.02, q=0.05, T=1.1)
    ts_pricer = TemperedStablePricer(**ts_params)
    t0 = time.time()
    price = ts_pricer.price(**option_params, N=20)
    t = time.time()
    print(price, f"Time: {t-t0}")
    return


if __name__ == "__main__":
    main()
