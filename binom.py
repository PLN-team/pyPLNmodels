import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
from pyPLNmodels import (
    Pln,
    PlnPCA,
    PlnPCAcollection,
    get_real_count_data,
    get_simulated_count_data,
    BIG,
    BIGN,
)

N = 1
nb_max = 50
counts, covariates, offsets, cov, coef = get_simulated_count_data(
    distrib = "BIGN", N_param=N, return_true_param=True, nb_cov = 0, no_offsets = True,
)

df = pd.DataFrame(counts.numpy())
df.to_csv("~/counts_big_0.csv", index = False)
df_sigma = pd.DataFrame(cov.numpy())
df_sigma.to_csv("~/sigma.csv", index = False)

# sns.heatmap(cov)
# plt.show()
bign = BIGN(counts, N_param=N, covariates = covariates, offsets = offsets)
bign.fit(nb_max_iteration=nb_max, tol=0)
# bign.show()
sns.scatterplot(x = cov.ravel(), y = cov.ravel()*0)
sns.scatterplot(x = cov.ravel(), y = bign.covariance.ravel())
sns.scatterplot(x = cov.ravel(), y = cov.ravel())

plt.show()
# print(bign)
# big = BIG(counts)
# big.fit(nb_max_iteration=nb_max, tol=0)
# big.show()
# print(big)
