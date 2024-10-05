#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Include for STL support (like std::vector)
#include <gsl/gsl_sf_gamma.h>  // Include GSL gamma functions
#include <vector>

namespace py = pybind11;

// Function to compute the upper incomplete gamma function for a vector s_vec and a 2D matrix x_matrix
std::vector<std::vector<std::vector<double>>> gamma_upper_incomplete(
    const std::vector<double>& s_vec, 
    const std::vector<std::vector<double>>& x_matrix) 
{
    size_t n1 = s_vec.size();                          // Size of the first dimension (vector s)
    size_t n2 = x_matrix.size();                       // Number of rows in the matrix (n2)
    size_t n3 = x_matrix[0].size();                    // Number of columns in the matrix (n3)

    // Resulting 3D matrix to store the gamma values
    std::vector<std::vector<std::vector<double>>> results(n1, std::vector<std::vector<double>>(n2, std::vector<double>(n3)));

    for (size_t i = 0; i < n1; ++i) {                  // Iterate over s_vec (n1 elements)
        for (size_t j = 0; j < n2; ++j) {              // Iterate over x_matrix rows (n2 elements)
            for (size_t k = 0; k < n3; ++k) {          // Iterate over x_matrix columns (n3 elements)
                // Compute the upper incomplete gamma function for each combination of s_vec[i] and x_matrix[j][k]
                results[i][j][k] = gsl_sf_gamma_inc(s_vec[i], x_matrix[j][k]);  // Access the scalar element in x_matrix
            }
        }
    }

    return results;  // Return the 3D matrix of results
}

// Interface for Python with pybind11
PYBIND11_MODULE(gamma_module, m) {
    m.def("gamma_upper_incomplete", &gamma_upper_incomplete, 
          "Compute upper incomplete gamma function Γ(s, x) for a vector and a 2D matrix using GSL");
}
