#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Inclure le support STL (comme std::vector)
#include <gsl/gsl_sf_gamma.h>  // Inclure les fonctions gamma de GSL
#include <vector>

namespace py = pybind11;

// Fonction pour calculer la fonction gamma incomplète supérieure pour les vecteurs s et x
std::vector<double> gamma_upper_incomplete(const std::vector<double>& s_vec, const std::vector<double>& x_vec) {
    if (s_vec.size() != x_vec.size()) {
        throw std::invalid_argument("Les vecteurs d'entrée doivent avoir la même longueur.");
    }

    std::vector<double> results;
    results.reserve(s_vec.size());

    for (size_t i = 0; i < s_vec.size(); ++i) {
        results.push_back(gsl_sf_gamma_inc(s_vec[i], x_vec[i]));
    }

    return results;
}

// Fonction pour calculer la fonction gamma incomplète inférieure non normalisée pour les vecteurs s et x
std::vector<double> gamma_lower_incomplete_non_normalized(const std::vector<double>& s_vec, const std::vector<double>& x_vec) {
    if (s_vec.size() != x_vec.size()) {
        throw std::invalid_argument("Les vecteurs d'entrée doivent avoir la même longueur.");
    }

    std::vector<double> results;
    results.reserve(s_vec.size());

    for (size_t i = 0; i < s_vec.size(); ++i) {
        double gamma_val = gsl_sf_gamma(s_vec[i]);
        double gamma_inc_P = gsl_sf_gamma_inc_P(s_vec[i], x_vec[i]);
        results.push_back(gamma_val * gamma_inc_P);
    }

    return results;
}

PYBIND11_MODULE(gamma_incomp, m) {
    m.def("gamma_upper_incomplete", &gamma_upper_incomplete, "Calculer la fonction gamma incomplète supérieure Γ(s, x) pour les vecteurs en utilisant GSL avec support pour les arguments négatifs");
    m.def("gamma_lower_incomplete_non_normalized", &gamma_lower_incomplete_non_normalized, "Calculer la fonction gamma incomplète inférieure non normalisée γ(s, x) pour les vecteurs en utilisant GSL");
}
