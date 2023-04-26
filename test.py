from pyPLNmodels.models import PLNPCA, _PLNPCA, PLN, ZIPLN
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir("./pyPLNmodels/")


(
    counts,
    covariates,
    offsets,
    true_Sigma,
    true_beta,
    true_infla,
) = get_simulated_count_data(return_true_param=True, n_samples=800)

# pln = ZIPLN(true_infla=true_infla, true_covariance=true_Sigma, true_coef=true_beta)
# pln.fit(counts, covariates, offsets, nb_max_iteration=250)
# pln.print_mse()
# sns.heatmap(pln._covariance.detach())
# plt.show()
# sns.heatmap(true_Sigma)
# plt.show()
full = _PLNPCA(rank=8)
full.fit(counts, covariates, offsets)
# print(full)
full.show()
