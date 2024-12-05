#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <gsl/gsl_sf_gamma.h>
#include <limits>  // Pour std::numeric_limits
#include <cmath>   // Pour std::isfinite, std::floor

namespace py = pybind11;

// Fonction pour calculer la gamma inférieure non normalisée pour un tenseur
py::array_t<double> gamma_lower_incomplete_tensor(const py::array_t<double>& a, const py::array_t<double>& x) {
    // Vérification que les dimensions des deux tenseurs sont compatibles
    py::buffer_info a_info = a.request();
    py::buffer_info x_info = x.request();

    if (a_info.ndim != x_info.ndim) {
        throw std::invalid_argument("Les tenseurs a et x doivent avoir le même nombre de dimensions.");
    }

    for (py::ssize_t i = 0; i < a_info.ndim; ++i) {
        if (a_info.shape[i] != x_info.shape[i]) {
            throw std::invalid_argument("Les tenseurs a et x doivent avoir les mêmes dimensions.");
        }
    }

    // Création d'un tableau de sortie avec les mêmes dimensions que `a`
    auto result = py::array_t<double>(a_info.shape);
    py::buffer_info result_info = result.request();

    // Accès direct aux données d'entrée et de sortie
    double* a_ptr = static_cast<double*>(a_info.ptr);
    double* x_ptr = static_cast<double*>(x_info.ptr);
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Calcul du nombre total d'éléments
    size_t total_elements = 1;
    for (py::ssize_t i = 0; i < a_info.ndim; ++i) {
        total_elements *= a_info.shape[i];
    }

    // Calcul de gamma inférieure non normalisée pour chaque élément
    for (size_t idx = 0; idx < total_elements; ++idx) {
        // Vérifier si `a[idx]` est un entier négatif ou zéro
        if (a_ptr[idx] <= 0 && std::floor(a_ptr[idx]) == a_ptr[idx]) {
            result_ptr[idx] = std::numeric_limits<double>::quiet_NaN();  // Assigner NaN
        } else {
            // Calculer Γ(s) et Γ(s, x)
            double gamma_val = gsl_sf_gamma(a_ptr[idx]);            // Γ(s)
            double gamma_upper = gsl_sf_gamma_inc(a_ptr[idx], x_ptr[idx]);  // Γ(s, x)

            // Vérifier si le résultat est valide
            double gamma_lower = gamma_val - gamma_upper;           // γ(s, x)
            if (!std::isfinite(gamma_lower)) {  // Vérifie si le résultat est infini ou NaN
                gamma_lower = std::numeric_limits<double>::quiet_NaN();
            }

            result_ptr[idx] = gamma_lower;
        }
    }

    return result;
}

PYBIND11_MODULE(gamma_incomp, m) {
    m.def("gamma_lower_incomplete_tensor", &gamma_lower_incomplete_tensor, 
          "Calculer la fonction gamma incomplète inférieure non normalisée γ(s, x) pour des tenseurs a et x en utilisant GSL, en remplaçant les résultats invalides par NaN");
}
