#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>    // Pour std::pow, std::fabs
#include <limits>   // Pour std::numeric_limits

namespace py = pybind11;

// Fonction pour calculer γ(a, x) en utilisant l'expansion donnée
double gamma_lower_series(double a, double x, size_t max_terms = 100, double tolerance = 1e-10) {
    if (x < 0) {
        throw std::invalid_argument("x doit être positif pour cette méthode.");
    }
    if (a <= 0 && std::floor(a) == a) {
        // Cas où a est un entier négatif ou zéro : résultat non défini
        return std::numeric_limits<double>::quiet_NaN();
    }

    double sum = 0.0;   // Somme de la série
    double term = std::pow(x, a);  // Premier terme de la série
    sum += term / a;    // Ajouter le premier terme

    for (size_t k = 1; k < max_terms; ++k) {
        term *= (-1.0) * x / k;  // Calcul du terme suivant
        sum += term / (a + k);   // Ajouter à la somme

        // Si la contribution devient négligeable, arrêter
        if (std::fabs(term) < tolerance) {
            break;
        }
    }

    return sum;  // Retourner la somme calculée
}

// Fonction pour calculer γ(a, x) pour un tableau de valeurs a et un scalaire x
py::array_t<double> gamma_lower_incomplete_non_normalized(
    const py::array_t<double>& a_array,
    double x,
    size_t max_terms = 100,
    double tolerance = 1e-10
) {
    // Vérification des dimensions du tableau
    py::buffer_info a_info = a_array.request();

    // Création du tableau de sortie avec les mêmes dimensions
    auto result_array = py::array_t<double>(a_info.shape);
    py::buffer_info result_info = result_array.request();

    // Accès aux données des tableaux
    double* a_ptr = static_cast<double*>(a_info.ptr);
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Calcul pour chaque élément
    size_t total_elements = 1;
    for (const auto& dim : a_info.shape) {
        total_elements *= dim;  // Nombre total d'éléments
    }

    for (size_t idx = 0; idx < total_elements; ++idx) {
        result_ptr[idx] = gamma_lower_series(a_ptr[idx], x, max_terms, tolerance);  // Calcul pour chaque élément
    }

    return result_array;  // Retourner le tableau de résultats
}

PYBIND11_MODULE(gamma_incomp, m) {
    m.def(
        "gamma_lower_incomplete_non_normalized",
        &gamma_lower_incomplete_non_normalized, 
        py::arg("a_array"), 
        py::arg("x"), 
        py::arg("max_terms") = 100, 
        py::arg("tolerance") = 1e-10,
        R"docstring(
        Calculer la fonction gamma incomplète inférieure non normalisée γ(a, x) pour un tableau de paramètres a et un scalaire x.

        Arguments:
        ----------
        - a_array : numpy.ndarray
            Tableau de paramètres a.
        - x : float
            Scalaire x utilisé dans le calcul.
        - max_terms : int, optionnel (par défaut 100)
            Nombre maximal de termes dans l'expansion en série.
        - tolerance : float, optionnel (par défaut 1e-10)
            Tolérance utilisée pour arrêter l'expansion en série lorsque les termes deviennent négligeables.

        Retourne:
        ---------
        numpy.ndarray :
            Tableau contenant les valeurs calculées de γ(a, x) pour chaque élément de a_array.
        )docstring"
    );
}
