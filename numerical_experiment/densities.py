import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scienceplots

from src.mellin_ts.densities.tsdensity import TSDensity

plt.style.use("science")

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_mellin_dens(
    density: TSDensity, range_n: list[int], x: npt.NDArray[np.float64]
) -> dict:
    dens = {}
    for n in range_n:
        print(f"Mellin with order {n}")
        dens[n] = density.density_mellin(x, n=n)
    return dens


def plot_mellin(
    ax: plt.Axes, densities: dict, x: npt.NDArray[np.float64], threshold=1e2
):
    n_max = max(list(densities.keys()))
    for n, dens in densities.items():
        # Identify regions where the absolute value of the function exceeds the threshold
        above_threshold = (dens > threshold) | (dens < 0)

        # Identify the last valid point before exceeding the threshold
        start_transitions = np.where(
            (~above_threshold[:-1]) & above_threshold[1:])[0]
        # Identify the first valid point after coming back below the threshold
        end_transitions = np.where(
            (above_threshold[:-1]) & ~above_threshold[1:])[0]

        # Mask y values that exceed the threshold
        y_masked = np.copy(dens)
        y_masked[above_threshold] = np.nan

        # Plot the function
        ax.scatter(
            x,
            y_masked,
            marker="x",
            label=rf"Series expansion $n={n}$",
            color="black",
            alpha=n / n_max,
        )
        # print(print(np.argwhere(y_masked < 0)))

        # Add red points at the last valid values before exceeding the threshold
        for i in start_transitions:
            ax.plot(
                x[i],
                dens[i],
                "ro",
                markersize=3,
            )
        # Add red points at the first valid values after dropping below the threshold
        for i in end_transitions:
            ax.plot(
                x[i + 1],
                dens[i + 1],
                "ro",
                markersize=3,
            )
        # ax.plot(
        #     x,
        #     dens,
        #     "b",
        #     label=rf"Series expansion $n={n}$",
        #     alpha=n / n_max,
        # )


def plot_fourier(ax1: plt.Axes, density: TSDensity, x: npt.NDArray[np.float64]):
    x_fourier = np.arange(np.min(x), np.max(x), 5e-2)
    print("Inverting Fourier...")
    dens_fourier = density.density_fourier(x_fourier, bounds=1e3, du=1e-1)
    print("Fourier inverted!")
    ax1.plot(
        x_fourier, dens_fourier, color="green", alpha=1, label="Fourier inversion"
    )


def plot_std_zones(ax: plt.Axes, density: TSDensity, n_std: int = 3):
    std = density.std
    mean = density.mean
    range_sigma = np.arange(-n_std, n_std + 1)
    xsigma = mean + std * range_sigma
    ax.set_xticks(xsigma)  # Placement des mu + i*sigma
    labels = get_labels(range_sigma)
    ax.set_xticklabels(
        labels,
        fontsize=8,
        rotation=45,
    )


def get_labels(range_sigma: npt.NDArray[np.float64]):
    labels = []
    for i in range_sigma:
        if i == 0:
            labels.append(r"$\mu$")
        elif i > 0:
            labels.append(rf"$\mu+{i}\sigma$")
        elif i < 0:
            labels.append(rf"$\mu{i}\sigma$")
    return labels


def build_figure(density: TSDensity, range_n: list[int], x: npt.NDArray[np.float64]):
    densities = get_mellin_dens(density, range_n, x)
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot()
    ax1.set_xlim(min(x), max(x) - 0.1)
    ax1.set_xlim(min(x), 5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(r"$x$")
    ax2 = ax1.twiny()
    ax2.tick_params(axis="x", length=0)
    ax2.set_yticks([])
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.grid()
    plot_fourier(ax1, density, x)
    plot_mellin(ax1, densities, x)
    plot_std_zones(ax2, density)
    ax1.legend(loc="upper left")
    plt.savefig("numerical_experiment/output/densities.png", dpi=600)
    plt.close()


def main():
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.35,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
    )
    densTS = TSDensity(**ts_params)
    x = np.arange(-6., 6, .05)
    # x = x[np.abs(x) > 1e-1]
    range_n = [60]
    build_figure(densTS, range_n, x)


if __name__ == "__main__":
    main()
