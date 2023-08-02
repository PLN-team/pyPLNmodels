import seaborn as sns
import matplotlib.pyplot as plt
import torch
from pyPLNmodels import (
    Pln,
    PlnPCA,
    PlnPCAcollection,
    get_real_count_data,
    get_simulated_count_data,
    BIG,
    BIGN,
)

N = 1000
counts, cov, offsets, cov, coef = get_simulated_count_data(
    distrib="BIGN", N_param=N, return_true_param=True
)
print("N:", N)
sns.heatmap(cov)
plt.show()
bign = BIGN(counts, N_param=N)
bign.fit(nb_max_iteration=1500, tol=0)
bign.show()
