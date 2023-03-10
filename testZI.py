import time

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyPLNmodels._utils import sample_PLN
from pyPLNmodels import ZIPLN, PLN
import pandas as pd
from scipy.special import lambertw as lw

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

n = 100
p = 20
q = 10
d = 1

# true_beta = torch.randn(d, p, device=DEVICE)
# B_zero = torch.randn(d,p, device = DEVICE)
# C = torch.randn(p, q, device=DEVICE)/5
# O = torch.ones((n, p), device=DEVICE)/2
# covariates = torch.ones((n, d), device=DEVICE)*0 + 1
# true_Sigma = torch.matmul(C,C.T)
# Y, _, _ = sample_PLN(C, true_beta, covariates, O,B_zero = B_zero)


# Y = pd.read_csv("./example_data/test_data/Y_test.csv")
# covariates = pd.read_csv("./example_data/test_data/cov_test.csv")
# O = pd.read_csv("./example_data/test_data/O_test.csv")

# zi = ZIPLN()
# zi.fit(Y,covariates,O, do_smart_init= False, nb_max_iteration=1000)
# zi.show()
## not working, we should at least get something not far from beta and sigma

# pln = PLN()
# pln.fit(Y,covariates, O)
# pln.show()

sigma = 1


def phi(mu, sigma):
    return np.exp(
        -1 / (2 * sigma**2) * (up_left_part(mu, sigma) + up_right_part(mu, sigma))
    ) / down_part(mu, sigma)


def up_left_part(mu, sigma):
    return lw(sigma**2 * np.exp(mu)) ** 2


def up_right_part(mu, sigma):
    return 2 * lw(sigma**2 * np.exp(mu))


def down_part(mu, sigma):
    return np.sqrt(1 + lw(sigma**2 * np.exp(mu)))


abscisse = np.linspace(-10, 10)
plt.plot(abscisse, phi(abscisse, sigma), label=r"$\varphi(x,1)$")
plt.plot(abscisse, 1 / (1 + np.exp(abscisse)), label=r"$\sigma(-x)$")
plt.legend()
plt.show()
