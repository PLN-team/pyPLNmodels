from pyPLNmodels.models import PlnPCAcollection, PlnPCA, Pln, ZIPln
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir("./pyPLNmodels/")


(
    endog,
    exog,
    offsets,
    true_Sigma,
    true_beta,
    true_infla,
) = get_simulated_count_data(return_true_param=True, n_samples=200, zero_inflated=True)

# pln = ZIPLN(true_infla=true_infla, true_covariance=true_Sigma, true_coef=true_beta)
# pln.fit(counts, covariates, offsets, nb_max_iteration=250)
# pln.print_mse()
# sns.heatmap(pln._covariance.detach())
# plt.show()
# sns.heatmap(true_Sigma)
# plt.show()
full = ZIPln(endog=endog, exog=exog, offsets=offsets)
full.fit(tol=0.001)
# print(full)
full.show()
