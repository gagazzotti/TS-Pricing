"""importing modules"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scienceplots
from fypy.model.levy.TemperedStable import TemperedStable
from fypy.pricing.fourier.HilbertEuropeanPricer import HilbertEuropeanPricer
from fypy.pricing.fourier.LewisEuropeanPricer import LewisEuropeanPricer
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward

from src.mellin_ts.pricers.OneSidedTSPricer import OneSidedTemperedStablePricer
from src.mellin_ts.pricers.TSPricer import TemperedStablePricer

plt.style.use(["science"])


def main():
    """Testing the convergence of the pricer"""
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        # alpha_m=0.35,
        # beta_m=0.5 - np.pi / 100,
        # lambda_m=0.4,
    )
    # option parameters
    strike = 1.5
    ttm = 10 ** (np.arange(-3, 0, 0.1))
    option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)
    # get the different prices
    lewis_prices = get_lewis_prices(option_params, ts_params)
    print("Lewis done!")
    mellin_prices = get_mellin_prices(option_params, ts_params)
    print("Mellin done!")
    ref_prices = get_ref_prices(option_params, ts_params)
    # final plot
    print(lewis_prices)
    print(mellin_prices)
    print(ref_prices)
    plot_difference(mellin_prices, lewis_prices, ref_prices, ttm)


def plot_difference(
    mellin_prices: npt.NDArray[np.float64],
    lewis_prices: npt.NDArray[np.float64],
    ref_prices: npt.NDArray[np.float64],
    ttm: npt.NDArray[np.float64],
):
    """Save the plot of short time pricing

    Args:
        mellin_prices (npt.NDArray[np.float64]): mellin prices
        lewis_prices (npt.NDArray[np.float64]): PROJ prices
        ref_prices (npt.NDArray[np.float64]): ref prices
        ttm (npt.NDArray[np.float64]): time
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(
        ttm,
        np.abs(ref_prices - lewis_prices) / ref_prices,
        marker="x",
        color="orange",
        label="Lewis",
    )
    plt.scatter(
        ttm,
        np.abs(ref_prices - mellin_prices) / ref_prices,
        marker="x",
        color="green",
        label="Mellin",
    )

    plt.xlabel(r"$T$")
    plt.ylabel("Relative error")
    plt.grid()
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("output/mellin_vs_proj.png", dpi=200)
    plt.close()

    return


def get_lewis_prices(option_params: dict, ts_params: dict):
    """_summary_

    Args:
        option_params (dict): option params
        ts_params (dict): TS params

    Returns:
        _type_: _description_
    """
    disc_curve = DiscountCurve_ConstRate(rate=option_params["r"])
    div_disc = DiscountCurve_ConstRate(rate=option_params["q"])
    fwd = EquityForward(
        S0=option_params["S0"], discount=disc_curve, divDiscount=div_disc
    )
    model = TemperedStable(
        forwardCurve=fwd, discountCurve=disc_curve, **ts_params)
    lewis_pricer = LewisEuropeanPricer(model=model)
    # lewis_pricer = ProjEuropeanPricer(model=model, N=2**16, order=3)
    # lewis_pricer = HilbertEuropeanPricer(model=model, N=2**17, Nh=2**8)

    lewis_prices = []
    for maturity in option_params["ttm"]:
        price = lewis_pricer.price(
            T=maturity, K=option_params["K"], is_call=True)
        lewis_prices.append(price)
    return np.array(lewis_prices)


def get_ref_prices(option_params: dict, ts_params: dict):
    """_summary_

    Args:
        option_params (dict): option params
        ts_params (dict): TS params

    Returns:
        _type_: _description_
    """
    disc_curve = DiscountCurve_ConstRate(rate=option_params["r"])
    div_disc = DiscountCurve_ConstRate(rate=option_params["q"])
    fwd = EquityForward(
        S0=option_params["S0"], discount=disc_curve, divDiscount=div_disc
    )
    model = TemperedStable(
        forwardCurve=fwd, discountCurve=disc_curve, **ts_params)
    proj_pricer = ProjEuropeanPricer(
        model=model, N=2**20, order=3, alpha_override=500)
    # proj_pricer = HilbertEuropeanPricer(model=model, N=2**15, Nh=2**7)
    # proj_pricer = LewisEuropeanPricer(model=model)
    proj_prices = []
    for maturity in option_params["ttm"]:
        price = proj_pricer.price(
            T=maturity, K=option_params["K"], is_call=True)
        proj_prices.append(price)
    return np.array(proj_prices)


def get_mellin_prices(option_params: dict, ts_params: dict):
    """_summary_

    Args:
        option_params (dict): option params
        ts_params (dict): TS params

    Returns:
        _type_: _description_
    """
    # ts_pricer = TemperedStablePricer(**ts_params)
    ts_pricer = OneSidedTemperedStablePricer(**ts_params)

    mellin_price = ts_pricer.price(**option_params, N=30)
    return mellin_price


if __name__ == "__main__":
    main()
