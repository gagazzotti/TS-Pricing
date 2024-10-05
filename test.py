import gamma_module
import numpy as np

# Vecteur s_vec de taille n1
s_vec = [-2.5, -3.0, 3]

# Matrice 2D x_matrix de taille n2 x n3
x_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

# Calcul de la fonction gamma incomplète
results = gamma_module.gamma_upper_incomplete(s_vec, x_matrix)

# Affichage des résultats
print("Results:", np.array(results), np.array(results).shape)
