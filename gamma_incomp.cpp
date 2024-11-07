#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // Support pour les arrays NumPy
#include <gsl/gsl_sf_gamma.h>  // Inclure les fonctions gamma de GSL
#include <stdexcept>           // Pour les exceptions

namespace py = pybind11;

// Fonction pour calculer la fonction gamma incomplète supérieure pour les arrays s et x de dimension arbitraire
py::array_t<double> gamma_upper_incomplete(py::array_t<double> s_array, py::array_t<double> x_array) {
    // Vérifier si les arrays ont les mêmes dimensions
    if (s_array.ndim() != x_array.ndim()) {
        throw std::invalid_argument("Les arrays d'entrée doivent avoir le même nombre de dimensions.");
    }
    for (ssize_t i = 0; i < s_array.ndim(); ++i) {
        if (s_array.shape(i) != x_array.shape(i)) {
            throw std::invalid_argument("Les arrays d'entrée doivent avoir les mêmes dimensions.");
        }
    }

    // Créer un array pour les résultats avec les mêmes dimensions que les arrays d'entrée
    py::array_t<double> result(s_array.shape(), s_array.shape() + s_array.ndim());

    // Obtenir des pointeurs pour accéder aux données des arrays
    auto s = s_array.unchecked<double>();
    auto x = x_array.unchecked<double>();
    auto res = result.mutable_unchecked<double>();

    // Parcourir tous les éléments en utilisant un index multi-dimensionnel
    py::ssize_t total_elements = s_array.size();
    for (py::ssize_t i = 0; i < total_elements; ++i) {
        res.data()[i] = gsl_sf_gamma_inc(s.data()[i], x.data()[i]);
    }

    return result;  // Retourner l'array des résultats
}

// Interface pour Python avec pybind11
PYBIND11_MODULE(gamma_module, m) {
    m.def("gamma_upper_incomplete", &gamma_upper_incomplete, "Calculer la fonction gamma incomplète supérieure Γ(s, x) pour des arrays de même dimension en utilisant GSL");
}
