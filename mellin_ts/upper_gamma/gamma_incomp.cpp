#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Inclure le support STL (comme std::vector)
#include <gsl/gsl_sf_gamma.h>  // Inclure les fonctions gamma de GSL
#include <vector>

namespace py = pybind11;

// Fonction pour calculer la fonction gamma incomplète supérieure pour les vecteurs s et x
std::vector<double> gamma_upper_incomplete(const std::vector<double>& s_vec, const std::vector<double>& x_vec) {
    // Vérifier si les vecteurs d'entrée ont la même longueur
    if (s_vec.size() != x_vec.size()) {
        throw std::invalid_argument("Les vecteurs d'entrée doivent avoir la même longueur.");
    }

    std::vector<double> results;
    results.reserve(s_vec.size()); // Réserver de l'espace pour les résultats

    for (size_t i = 0; i < s_vec.size(); ++i) {
        // Calculer la fonction gamma incomplète supérieure pour chaque paire (s, x)
        results.push_back(gsl_sf_gamma_inc(s_vec[i], x_vec[i])); // Utiliser gsl_sf_gamma_inc
    }

    return results; // Retourner le vecteur des résultats
}

// Interface pour Python avec pybind11
PYBIND11_MODULE(gamma_incomp, m) {
    m.def("gamma_upper_incomplete", &gamma_upper_incomplete, "Calculer la fonction gamma incomplète supérieure Γ(s, x) pour les vecteurs en utilisant GSL avec support pour les arguments négatifs");
}
