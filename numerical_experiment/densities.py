import warnings
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from mellin_ts.densities.TSDensity import TSDensity

plt.style.use("science")

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.1 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.65,
        beta_m=0.5 - np.pi / 100,
        lambda_m=0.4,
    )
    ts_params = dict(
        alpha_p=0.44,
        beta_p=0.2 + np.exp(1) / 10,
        lambda_p=1.4,
        alpha_m=0.64,
        beta_m=0.1 + np.exp(1) / 10,
        lambda_m=0.4,
    )
    densTS = TSDensity(**ts_params)

    # Définition des points pour les densités Mellin
    x = np.arange(-7.55, 7.6, 5 * 1e-1)
    density_Mellin_40 = densTS.density_Mellin(x, n=20)
    density_Mellin_50 = densTS.density_Mellin(x, n=30)
    density_Mellin_55 = densTS.density_Mellin(x, n=55)

    # Définition des points pour la densité de Fourier
    x_fourier = np.arange(min(x), max(x), 1e-2)
    density_Fourier = densTS.density_Fourier(x_fourier, bounds=1e3, du=1e-1)

    # Plot des densités Mellin et Fourier
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    plt.xlim(min(x_fourier), max(x_fourier) - 0.1)

    # Densités Mellin avec différentes valeurs de n
    ax1.scatter(
        x,
        density_Mellin_40,
        marker="d",
        color="black",
        label=r"Series expansion $n=20$",
    )
    ax1.scatter(
        x,
        density_Mellin_50,
        marker="o",
        color="black",
        label=r"Series expansion $n=30$",
    )
    ax1.scatter(
        x,
        density_Mellin_55,
        marker="x",
        color="black",
        label=r"Series expansion $n=55$",
    )
    ax1.set_xlabel(r"$x$")

    # Densité Fourier
    ax1.plot(x_fourier, density_Fourier, color="blue", label="Fourier inversion")

    # Ajout des intervalles à ±1σ, ±2σ, ±3σ
    std = densTS.std
    mean = densTS.mean

    mask_1std = (x_fourier >= mean - std) & (x_fourier <= mean + std)
    mask_2std = ((x_fourier >= mean - 2 * std) & (x_fourier <= mean - std)) | (
        (x_fourier >= mean + std) & (x_fourier <= mean + 2 * std)
    )
    mask_3std = ((x_fourier >= mean - 3 * std) & (x_fourier <= mean - 2 * std)) | (
        (x_fourier >= mean + 2 * std) & (x_fourier <= mean + 3 * std)
    )
    mask_4std = ((x_fourier >= mean - 4 * std) & (x_fourier <= mean - 3 * std)) | (
        (x_fourier >= mean + 3 * std) & (x_fourier <= mean + 4 * std)
    )
    mask_5std = ((x_fourier >= mean - 5 * std) & (x_fourier <= mean - 4 * std)) | (
        (x_fourier >= mean + 4 * std) & (x_fourier <= mean + 5 * std)
    )
    mask_6std = ((x_fourier >= mean - 5 * std) & (x_fourier <= mean - 6 * std)) | (
        (x_fourier >= mean + 5 * std) & (x_fourier <= mean + 6 * std)
    )
    # Remplissage des intervalles
    ax2.set_ylim(-2, 2)
    ax2.fill_between(
        x_fourier,
        ax2.get_ylim()[0],  # Prendre la valeur minimale de l'axe y
        ax2.get_ylim()[1],  # Prendre la valeur maximale de l'axe y
        where=mask_1std,
        color="green",
        alpha=0.6,
        label=r"$[\mu-\sigma,\mu+\sigma]$",
    )
    ax2.fill_between(
        x_fourier,
        ax2.get_ylim()[0],  # Prendre la valeur minimale de l'axe y
        ax2.get_ylim()[1],  # Prendre la valeur maximale de l'axe y
        where=mask_2std,
        color="green",
        alpha=0.5,
        label=r"$[\mu-2\sigma,\mu+2\sigma]$",
    )
    ax2.fill_between(
        x_fourier,
        ax2.get_ylim()[0],  # Prendre la valeur minimale de l'axe y
        ax2.get_ylim()[1],  # Prendre la valeur maximale de l'axe y
        where=mask_3std,
        color="green",
        alpha=0.4,
        label=r"$[\mu-3\sigma,\mu+3\sigma]$",
    )
    ax2.fill_between(
        x_fourier,
        ax2.get_ylim()[0],  # Prendre la valeur minimale de l'axe y
        ax2.get_ylim()[1],  # Prendre la valeur maximale de l'axe y
        where=mask_4std,
        color="green",
        alpha=0.3,
        label=r"$[\mu-4\sigma,\mu+4\sigma]$",
    )
    ax2.fill_between(
        x_fourier,
        ax2.get_ylim()[0],  # Prendre la valeur minimale de l'axe y
        ax2.get_ylim()[1],  # Prendre la valeur maximale de l'axe y
        where=mask_5std,
        color="green",
        alpha=0.2,
        label=r"$[\mu-5\sigma,\mu+5\sigma]$",
    )
    ax2.set_yticks([])
    # Ajustements finaux
    plt.xlabel(r"$x$")
    ax1.grid()
    ax1.set_ylim(0, 0.6)

    # Première légende pour les séries Mellin et Fourier

    # Deuxième légende pour les intervalles
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.savefig("numerical_experiment/output/densities.png")
    plt.close()


if __name__ == "__main__":
    main()
