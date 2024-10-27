import mellin_ts.pricing.lower_gamma.gamma_incomp as gamma_incomp
import mpmath


s_vec = [1.0, 2.0, 3.0]
x_vec = [0.5, 1.5, 2.5]

upper_results = gamma_incomp.gamma_upper_incomplete(s_vec, x_vec)
lower_results = gamma_incomp.gamma_lower_incomplete_non_normalized(s_vec, x_vec)

print("Upper incomplete gamma results:", upper_results)
print("Lower incomplete gamma results:", lower_results)

print([mpmath.gamma(i) - mpmath.gammainc(i, j) for i, j in zip(s_vec, x_vec)])
