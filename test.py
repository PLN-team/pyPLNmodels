from pyPLNmodels.models import PLNPCA, _PLNPCA, PLN, ZIPLN
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import os

os.chdir("./pyPLNmodels/")


(
    counts,
    covariates,
    offsets,
    true_Sigma,
    true_beta,
    true_infla,
) = get_simulated_count_data(return_true_param=True)
