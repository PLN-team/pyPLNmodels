from pyPLNmodels.VEM import PLNPCA, PLN, _PLNPCA, _PLNPCA_noS
from pyPLNmodels._utils import get_simulated_count_data, get_real_count_data
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
os.chdir("pyPLNmodels/")

Y = get_real_count_data()
q = 5
lr = 0.01
# nb_max = 1000
pca = _PLNPCA(q)
pca.fit(Y, tol=0.00001, lr=lr)
pca.show()

# pca = _PLNPCA(q)
# pca.fit(Y, tol = 0.000000001)
# pca.show()
nospca = _PLNPCA_noS(q)
nospca.fit(Y, tol=0.00001, lr=lr)
nospca.show()
abscisse_pca = np.arange(len(pca.elbos_list))
abscisse_nospca = np.arange(len(nospca.elbos_list))
plt.plot(abscisse_pca, pca.elbos_list, label="pca")
plt.plot(abscisse_nospca, nospca.elbos_list, label="nospca")
plt.legend()
plt.show()
