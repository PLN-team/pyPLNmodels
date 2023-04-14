import time
import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyPLNmodels._utils import sample_PLN, get_real_count_data
from pyPLNmodels import PLNPCA, PLN
import pandas as pd

print("dir", os.getcwd())
# os.chdir("/pyPLNmodels")
n = 1000
p = 1000
q = 40
d = 1


if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

ranks = [10]  # , 50, 100, 200,400,700,1000]
# ranks = [5,10]
def get_list_iteration(pca):
    return [model.nb_iteration_done for model in pca.dict_PLNPCA.values()]


def launch_some_iterations(number_of_iterations, tol):
    list_mean_iteration = {
        "tril": np.zeros(len(ranks)),
        "notril": np.zeros(len(ranks)),
        "doubletril": np.zeros(len(ranks)),
    }
    list_mean_elbos = {
        "tril": np.zeros(len(ranks)),
        "notril": np.zeros(len(ranks)),
        "doubletril": np.zeros(len(ranks)),
    }
    for _ in range(number_of_iterations):
        # true_beta = torch.randn(d, p, device=DEVICE)
        # C = torch.randn(p, q, device=DEVICE)/5
        # O = torch.ones((n, p), device=DEVICE)/2
        # covariates = torch.ones((n, d), device=DEVICE)*0 + 1
        # true_Sigma = torch.matmul(C,C.T)
        # Y, _, _ = sample_PLN(C, true_beta, covariates, O)
        Y, _, _ = get_real_count_data()
        covariates = None
        O = None
        pca_tril = PLNPCA(ranks=ranks, tril_number=1)
        pca_tril.fit(Y, covariates, O)
        pca_notril = PLNPCA(ranks=ranks, tril_number=0)
        pca_notril.fit(Y, covariates, O, tol=tol)
        pca_doubletril = PLNPCA(ranks=ranks, tril_number=2)
        pca_doubletril.fit(Y, covariates, O, tol=tol)
        list_mean_iteration["tril"] += (
            np.array(get_list_iteration(pca_tril)) / number_of_iterations
        )
        list_mean_iteration["notril"] += (
            np.array(get_list_iteration(pca_notril)) / number_of_iterations
        )
        list_mean_iteration["doubletril"] += (
            np.array(get_list_iteration(pca_doubletril)) / number_of_iterations
        )
        list_mean_elbos["tril"] += (
            np.array(list(pca_tril.loglikes.values())) / number_of_iterations
        )
        list_mean_elbos["notril"] += (
            np.array(list(pca_notril.loglikes.values())) / number_of_iterations
        )
        list_mean_elbos["doubletril"] += (
            np.array(list(pca_doubletril.loglikes.values())) / number_of_iterations
        )
    return list_mean_iteration, list_mean_elbos


list_mean_iteration, list_mean_elbos = launch_some_iterations(5, tol=0.1)
fig, axes = plt.subplots(2)
axes[0].plot(ranks, list_mean_iteration["tril"], label="tril")
axes[0].plot(ranks, list_mean_iteration["notril"], label="notril")
axes[0].plot(ranks, list_mean_iteration["doubletril"], label="doubletril")
axes[0].legend()

axes[1].plot(ranks, list_mean_elbos["tril"], label="tril")
axes[1].plot(ranks, list_mean_elbos["notril"], label="notril")
axes[1].plot(ranks, list_mean_elbos["doubletril"], label="doubletril")
axes[1].legend()
plt.show()
# print(pca.best_model.Sigma)
# pln = PLN()
# pln.fit(Y, covariates, O)
# print("loglike pln", pln.loglike)
