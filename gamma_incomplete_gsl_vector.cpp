#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Include for STL support (like std::vector)
#include <gsl/gsl_sf_gamma.h>  // Include GSL gamma functions
#include <vector>

namespace py = pybind11;

// Function to compute the upper incomplete gamma function for vectors of s and x
std::vector<double> gamma_upper_incomplete(const std::vector<double>& s_vec, const std::vector<double>& x_vec) {
    // Check if input vectors are of the same length
    if (s_vec.size() != x_vec.size()) {
        throw std::invalid_argument("Input vectors must have the same length.");
    }

    std::vector<double> results;
    results.reserve(s_vec.size()); // Reserve space for the results

    for (size_t i = 0; i < s_vec.size(); ++i) {
        // Compute the upper incomplete gamma function for each pair (s, x)
        results.push_back(gsl_sf_gamma_inc(s_vec[i], x_vec[i])); // Use gsl_sf_gamma_inc
    }

    return results; // Return the vector of results
}

// Interface for Python with pybind11
PYBIND11_MODULE(gamma_module, m) {
    m.def("gamma_upper_incomplete", &gamma_upper_incomplete, "Compute upper incomplete gamma function Γ(s, x) for vectors using GSL with support for negative arguments");
}
