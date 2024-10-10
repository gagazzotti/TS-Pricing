import numpy as np
import matplotlib.pyplot as plt

# Paramètres de l'exemple
mu = 0  # Moyenne
sigma = 1  # Écart-type
x = np.linspace(-7, 7, 400)  # Intervalle sur l'axe des x

# Création de la figure et des axes
fig, ax = plt.subplots()

# Tracé des courbes (remplacez par vos propres données)
ax.plot(
    x, np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), label="Fourier inversion", color="blue"
)

# Ajout des points de série (remplacez par vos données)
n_values = [20, 30, 60]
x_series = [-6, -4, 0, 2]  # Exemple de valeurs x pour les points
y_series = [0.1, 0.2, 0.4, 0.3]  # Exemple de valeurs y pour les points

for n, marker in zip(n_values, ["d", "o", "x"]):
    ax.scatter(x_series, y_series, label=f"Series expansion n = {n}", marker=marker)

# Ajout des bandes d'intervalle de confiance en rouge (remplacez par vos propres valeurs)
# for i in range(1, 7):
#     ax.axvspan(mu - i * sigma, mu + i * sigma, color="red", alpha=0.1 * i)

# Ajout d'un second axe x en haut
ax2 = ax.twiny()

# Définir les positions des mu + i*sigma sur l'axe supérieur
sigma_values = np.arange(-6, 7, 1)  # De mu - 6*sigma à mu + 6*sigma
x_sigma = mu + sigma_values * sigma

ax2.set_xlim(ax.get_xlim())  # Synchronisation des limites
ax2.set_xticks(x_sigma)  # Placement des mu + i*sigma
ax2.set_xticklabels(
    [f"$\mu {i:+d}\sigma$" for i in sigma_values], fontsize=8, rotation=45, ha="right"
)  # Diagonale

# Supprimer les petites barres (ticks) sur l'axe du haut
ax2.tick_params(axis="x", length=0)  # Longueur des barres mise à 0

# Personnalisation de l'affichage
ax.set_xlabel("x")
ax2.grid()
# ax.grid()
ax.set_ylabel("Density")
ax.legend(loc="upper left")
ax2.set_xlabel(r"$\mu + i\sigma$")

# Affichage
plt.show()
