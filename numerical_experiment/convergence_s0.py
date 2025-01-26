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
    return np.log(option_params["S0"]/option_params["strike"]) + drift


def main():
    """Testing the convergence of the pricer"""
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.35,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
        # alpha_m=0.44,
        # beta_m=0.1 + np.exp(1) / 10,
        # lambda_m=1.4,
    )
    n_start, n_end = 1, 101
    range_n = np.arange(n_start, n_end, 5) - 1
    range_n[0] = 1
    range_n = list(range_n)
    ts_pricer = TemperedStablePricer(**ts_params)
    zeta = ts_pricer.zeta
    option_params = dict(S0=np.arange(0.5, 1.5, 0.1),
                         r=0.02, q=0.05, ttm=0.7, strike=1)
    # range_s0 = np.hstack((option_params["S0"], option_params["strike"]*np.exp(-(
    #     option_params["r"]-option_params["q"]+zeta)*option_params["ttm"])))
    # option_params["S0"] = np.sort(range_s0)
    moneyness = compute_moneyness(option_params, zeta)
    print("Moneyness")
    print(np.array(moneyness))
    proj_prices = get_proj_prices(option_params, ts_params)
    mellin_prices = get_mellin_prices(
        option_params, ts_pricer, [50])
    print(np.array(proj_prices))
    print(np.array(mellin_prices[50]))
    # print(option_params["S0"], mellin_prices)
    x_axis = option_params["S0"]

    plt.plot(x_axis, proj_prices, label="PROJ")
    for n, price_mellin in mellin_prices.items():
        plt.scatter(x_axis, price_mellin,
                    marker="x", label=fr"Mellin series $N={n}$")
    plt.plot(x_axis, (option_params["S0"]-option_params["strike"])
             * (option_params["S0"]-option_params["strike"] >= 0), label=r"$x\mapsto (x-K)^+$")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$k$")
    plt.ylim(0, 1)
    plt.show()


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
    proj_prices = []
    for s0 in tqdm.tqdm(option_params["S0"]):
        disc_curve = DiscountCurve_ConstRate(rate=option_params["r"])
        div_disc = DiscountCurve_ConstRate(rate=option_params["q"])
        fwd = EquityForward(
            S0=s0, discount=disc_curve, divDiscount=div_disc)
        model = TemperedStable(
            forwardCurve=fwd, discountCurve=disc_curve, **ts_params)
        proj_pricer = ProjEuropeanPricer(
            model=model, N=2**16, order=3)
        price = proj_pricer.price(
            option_params["ttm"], option_params["strike"], is_call=True)
        proj_prices.append(price)
    return proj_prices


def get_mellin_prices(
    option_params: dict,
    ts_pricer: TemperedStablePricer,
    range_n: np.ndarray
):
    """TBD

    Args:
        ts_pricer (TemperedStablePricer): pricer
        option_params (dict): standard params
        n_start (int, optional): n to begin with. Defaults to 10.
        n_end (int, optional): n to end with. Defaults to 16.

    Returns:
        prices
    """
    # PROJ framewrok
    mellin_prices = {}
    for n in range_n:
        mellin_prices_list = []
        for s0 in tqdm.tqdm(option_params["S0"]):
            try:
                price = ts_pricer.price(s0, option_params["strike"], option_params["r"],
                                        option_params["q"], option_params["ttm"], N=n)
            except NotImplementedError:
                price = np.nan
            mellin_prices_list.append(price)
        mellin_prices[n] = mellin_prices_list
    return mellin_prices


if __name__ == "__main__":
    main()
