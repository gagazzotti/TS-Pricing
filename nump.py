import numpy as np

N = 10
n1 = np.arange(N)[:, None, None, None, None]
n2 = np.arange(N)[None, :, None, None, None]
n3 = np.arange(N)[None, None, :, None, None]

gamma = np.zeros_like(n1 + n2 + n3)
gamma[0, 0, 0, :] = 1.1
print(gamma[0, 0, 0])

beta_p, beta_m = 0.5, 0.5
low_gamma_term = np.zeros_like(n1 + n2 + n3).astype(float)
print(low_gamma_term[0, 0, 0])
low_gamma_term[0, 0, 0, :] = beta_p / (beta_m + beta_p)
print("__", low_gamma_term[0, 0, 0], beta_p / (beta_m + beta_p))
