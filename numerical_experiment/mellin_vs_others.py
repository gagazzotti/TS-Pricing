"""importing modules"""

import time
import numpy as np
import pandas as pd
import tqdm

# import numpy.typing as npt
# import matplotlib.pyplot as plt
import scienceplots
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.levy.TemperedStable import TemperedStable


from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward

# from matplotlib.ticker import FuncFormatter

# import time

from mellin_ts.TemperedStablePricers import (
    OneSidedTemperedStablePricer,
    TemperedStablePricer,
)

# plt.style.use(["science"])
pd.set_option("display.precision", 8)


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
    ts_p_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
    )
    strike = 1.5
    ttm = 1.2
    eps_ts_p = 10.0 ** (-np.arange(5, 6))
    eps_ts = 10.0 ** (-np.arange(2, 4))
    option_params = dict(S0=1, K=strike, r=0.02, q=0.05, ttm=ttm)

    mellin_time_ts = get_time_mellin_ts(option_params, ts_params, eps_ts)
    proj_time_ts = get_time_PROJ_ts(option_params, ts_params, eps_ts, one_sided=False)
    df_ts = pd.concat([mellin_time_ts, proj_time_ts], axis=1)
    proj_time_ts_p = get_time_PROJ_ts(
        option_params, ts_p_params, eps_ts_p, one_sided=True
    )
    mellin_time_ts_p = get_time_mellin_ts_p(option_params, eps_ts_p)
    df_ts_p = pd.concat([mellin_time_ts_p, proj_time_ts_p], axis=1)
    print(10 * "#")
    print("Mellin TS Double-sided")
    print(10 * "#")
    print(df_ts)
    print("\n\n")
    print(10 * "#")
    print("Mellin TS One-sided")
    print(10 * "#")
    print(df_ts_p)
    save_results(df_ts, "TS")
    save_results(df_ts_p, "TSp")


def get_proj_curves(
    option_params: dict,
) -> tuple[EquityForward, DiscountCurve_ConstRate]:
    """generates usefull curves for PROJ method

    Args:
        option_params (dict): option params

    Returns:
        tuple[EquityForward, DiscountCurve_ConstRate]: two curves
    """
    disc_curve = DiscountCurve_ConstRate(rate=option_params["r"])
    div_disc = DiscountCurve_ConstRate(rate=option_params["q"])
    fwd = EquityForward(
        S0=option_params["S0"], discount=disc_curve, divDiscount=div_disc
    )
    return fwd, disc_curve


def get_time_PROJ_ts(option_params: dict, ts_params: dict, eps: list, one_sided=True):
    # print("-----")
    if one_sided:
        proj_price_ref = 0.18769860488549217
        alpha_list = [np.nan, 10, 12.5, 15, 17.5, 20]
    else:
        proj_price_ref = 0.22968572289948497
        alpha_list = [np.nan, 25, 30, 35, 40, 50]

    comp_time = {error: {str(alpha): 0 for alpha in alpha_list} for error in eps}
    fwd, disc_curve = get_proj_curves(option_params)
    model = TemperedStable(forwardCurve=fwd, discountCurve=disc_curve, **ts_params)
    logN_max = 20
    for error in tqdm.tqdm(eps):
        for alpha in alpha_list:
            logN = 2
            price = 0
            alpha_dic = str(alpha)
            while np.abs(price - proj_price_ref) > error and logN < logN_max:
                # print(logN)
                proj_pricer = ProjEuropeanPricer(
                    model=model, N=2**logN, order=3, alpha_override=alpha
                )
                t0 = time.time()
                price = proj_pricer.price(
                    T=option_params["ttm"], K=option_params["K"], is_call=True
                )
                t_comp = time.time() - t0
                comp_time[error][alpha_dic] = t_comp
                logN += 1

            if logN == logN_max:
                comp_time[error][alpha_dic] = None

    df = pd.DataFrame.from_dict(comp_time, orient="index")
    df.rename(columns={"nan": "Default"}, inplace=True)
    return df


def get_time_mellin_ts(option_params: dict, ts_params: dict, eps: dict):
    proj_price_ref = 0.22968572289948497
    ts_pricer = TemperedStablePricer(**ts_params)
    nmax = 100
    comp_time = {"Mellin": {error: 0 for error in eps}}
    for error in tqdm.tqdm(eps):
        n = 5
        price = 0
        while np.abs(price - proj_price_ref) > error and n < nmax:
            t0 = time.time()
            price = ts_pricer.price(**option_params, N=n)
            comp_time["Mellin"][error] = time.time() - t0
            n += 1
            # print(price, np.abs(price - proj_price_ref))
        if n == nmax:
            comp_time["Mellin"][error] = None
    df = pd.DataFrame.from_dict(comp_time)
    return df


def get_time_mellin_ts_p(option_params: dict, eps: dict):
    params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
    )
    proj_price_ref = 0.18769860488549217
    ts_pricer = OneSidedTemperedStablePricer(**params)
    # nmax = 60
    nmax = 60
    comp_time = {"Mellin": {error: 0 for error in eps}}
    for error in tqdm.tqdm(eps):
        n = 5
        price = 0
        while np.abs(price - proj_price_ref) > error and n < nmax:
            t0 = time.time()
            price = ts_pricer.price(**option_params, N=n)
            comp_time["Mellin"][error] = time.time() - t0
            n += 1
        if n == nmax:
            comp_time["Mellin"][error] = None
    df = pd.DataFrame.from_dict(comp_time)
    return df


def save_results(df: pd.DataFrame, title: str, latex: bool = True):
    # Sauvegarder le DataFrame en .txt
    path = "numerical_experiment/output/"
    txt_filename = f"{title}.txt"
    df.to_csv(path + txt_filename, sep="\t", index=False)
    print(f"DataFrame sauvegardé en {txt_filename}")

    # Sauvegarder en LaTeX si latex=True
    if latex:
        latex_filename = path + f"{title}.tex"
        with open(latex_filename, "w") as f:
            f.write(df.to_latex(index=False))
        print(f"DataFrame sauvegardé en {latex_filename}")


if __name__ == "__main__":
    main()
