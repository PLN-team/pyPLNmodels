from pyPLNmodels.models import PLNPCA, PLN, _PLNPCA, _PLNPCA_noS
from pyPLNmodels._utils import get_simulated_count_data, get_real_count_data
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
os.chdir("pyPLNmodels/")

Y = get_real_count_data()
q = 10


pln = PLN()
pln.fit(Y)
true_Sigma = pln.Sigma
true_beta = pln.beta

tol = 0.0000001

pca = _PLNPCA(q, true_Sigma=true_Sigma, true_beta=true_beta)
pca.fit(Y, tol=tol)

nos = _PLNPCA_noS(q, true_Sigma=true_Sigma, true_beta=true_beta)
nos.fit(Y, tol=tol)


y_nos_sigma = nos.mse_Sigma_list
y_nos_beta = nos.mse_beta_list

y_s_sigma = pca.mse_Sigma_list
y_s_beta = pca.mse_beta_list


abs_nos = np.arange(len(y_nos_sigma))
abs_s = np.arange(len(y_s_sigma))
plt.plot(abs_nos, y_nos_sigma, label="no S Sigma", color="blue", linestyle="--")
plt.plot(abs_nos, y_nos_beta, label="no S beta", color="red", linestyle="--")

plt.plot(abs_s, y_s_sigma, label="S Sigma", color="blue")
plt.plot(abs_s, y_s_beta, label="S beta", color="red")

plt.legend()
plt.show()
