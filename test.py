from pyPLNmodels.models import PlnPCAcollection, PlnPCA, Pln, ZIPln
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch

os.chdir("./pyPLNmodels/")


(
    endog,
    exog,
    offsets,
    true_Sigma,
    true_beta,
    true_infla,
) = get_simulated_count_data(
    return_true_param=True, n_samples=200, zero_inflated=True, nb_cov=1, add_const=True
)
# endog = get_real_count_data()
# endog = torch.from_numpy(endog)
# offsets = None
# exog = None
print("percentage zeros:", torch.sum(endog == 0) / (endog.shape[0] * endog.shape[1]))
# full = ZIPln(endog=endog, exog=exog, offsets=offsets)
full = ZIPln(endog, offsets=offsets, exog=exog, add_const=True)
full.fit(nb_max_iteration=200)
# print(full)
full.show()
# beta_zi = full._coef.detach()
print("mse:", torch.mean((full._coef - true_beta) ** 2))
print("mse:", torch.mean((full._coef_inflation - true_infla) ** 2))
sns.heatmap(true_Sigma * (1 - torch.eye(true_Sigma.shape[0])))
plt.show()


# pln = Pln(endog=endog)
# pln.fit()
# pln.show()
# print("mse :", torch.mean((pln._coef - beta_zi) ** 2))
