from pyPLNmodels.models import PlnPCAcollection, PlnPCA, Pln, ZIPln
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch

os.chdir("./pyPLNmodels/")


# (
#     endog,
#     exog,
#     offsets,
#     true_Sigma,
#     true_beta,
#     true_infla,
# ) = get_simulated_count_data(return_true_param=True, n_samples=200, zero_inflated=True)
endog = get_real_count_data()
endog = torch.from_numpy(endog)

print("percentage zeros:", torch.sum(endog == 0) / (endog.shape[0] * endog.shape[1]))
# full = ZIPln(endog=endog, exog=exog, offsets=offsets)
full = ZIPln(endog)
full.fit(nb_max_iteration=100)
# print(full)
full.show()
beta_zi = full._coef.detach()


pln = Pln(endog=endog)
pln.fit()
print("mse :", torch.mean((pln._coef - beta_zi) ** 2))
