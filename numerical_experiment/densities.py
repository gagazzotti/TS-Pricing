import warnings
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import scienceplots
from mellin_ts.densities.TSDensity import TSDensity

plt.style.use("science")

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_mellin_dens(
    density: TSDensity, range_n: list[int], x: npt.NDArray[np.float64]
) -> dict:
    dens = {}
    for n in range_n:
        dens[n] = density.density_Mellin(x, n=n)
    return dens


def plot_mellin(ax: plt.Axes, densities: dict, x: npt.NDArray[np.float64]):
    markers = ["d", "o", "x"]
    i = 0
    for n, dens in densities.items():
        ax.scatter(
            x,
            dens,
            marker=markers[i],
            color="black",
            label=rf"Series expansion $n={n}$",
        )
        i += 1


def plot_fourier(ax1: plt.Axes, density: TSDensity, x: npt.NDArray[np.float64]):
    x_fourier = np.arange(np.min(x), np.max(x), 1e-2)
    dens_fourier = density.density_Fourier(x_fourier, bounds=1e3, du=1e-1)
    ax1.plot(x_fourier, dens_fourier, color="blue", label="Fourier inversion")


def plot_std_zones(
    ax: plt.Axes, density: TSDensity, x: npt.NDArray[np.float64], n_std: int = 6
):
    std = density.std
    mean = density.mean
    x_zone = np.arange(np.min(x), np.max(x), 1e-2)
    for n in range(n_std):
        mask_std = ((x_zone >= mean - (n + 1) * std) & (x_zone <= mean - n * std)) | (
            (x_zone >= mean + n * std) & (x_zone <= mean + (n + 1) * std)
        )
        ax.fill_between(
            x_zone,
            ax.get_ylim()[0],  # Prendre la valeur minimale de l'axe y
            ax.get_ylim()[1],  # Prendre la valeur maximale de l'axe y
            where=mask_std,
            color="green",
            alpha=0.7 - n / 10,
            label=rf"$[\mu-{n+1}\sigma,\mu+{n+1}\sigma]$",
        )


def build_figure(density: TSDensity, range_n: list[int], x: npt.NDArray[np.float64]):
    densities = get_mellin_dens(density, range_n, x)
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    # limits
    ax1.set_xlim(min(x), max(x) - 0.1)
    ax1.set_ylim(0, 0.6)
    ax2.set_ylim(-2, 2)
    # label
    ax1.set_xlabel(r"$x$")
    # grid
    ax1.grid()
    # disable ax2 ticks
    ax2.set_yticks([])
    plot_mellin(ax1, densities, x)
    plot_fourier(ax1, density, x)
    plot_std_zones(ax2, density, x)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.savefig("numerical_experiment/output/densities.png", dpi=600)
    plt.close()


def main():
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.65,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
    )
    densTS = TSDensity(**ts_params)

    # Définition des points pour les densités Mellin
    x = np.arange(-7.55, 7.6, 5 * 1e-1)
    range_n = [20, 30, 60]
    build_figure(densTS, range_n, x)


if __name__ == "__main__":
    main()
