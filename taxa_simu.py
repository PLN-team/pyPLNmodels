import pandas as pd
import os
import numpy as np
import torch
from pyPLNmodels.models import Brute_ZIPln, Pln
from pyPLNmodels import (
    ZIPln,
    load_model,
    PlnPCA,
    Pln,
    load_microcosm,
    get_zipln_simulated_count_data,
)
import seaborn as sns
import matplotlib.pyplot as plt
import pkg_resources
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# print(sklearn.__version__)

endog, exog = load_microcosm(
    get_interaction=False, n_samples=2000, dim=1300, for_formula=False, min_perc=0.02
)

columns_site = [col for col in exog.columns if "site" in col]
columns_time = [col for col in exog.columns if "time" in col]

exog_site = exog[columns_site]
exog_time = exog[columns_time]

# endog, exog, exog_inflation, offsets  = get_zipln_simulated_count_data(zero_inflation_formula = "column-wise", nb_cov_inflation = 3)

dict_init = None
# zi = ZIPln.from_formula("endog ~ 1 + exog", data=data, dict_initialization=dict_init)
# zi = ZIPln(
#     data["endog"],
#     exog=data["exog"],
#     exog_inflation=data["exog"],
#     dict_initialization=dict_init,
# )
zi = ZIPln(endog, exog=exog, exog_inflation=exog, dict_initialization=dict_init)
pln = Pln(endog)
pln.fit()
pln.show()

# zi.fit(tol=0, nb_max_iteration=200, verbose=True)
# zi.show()
# brute = Brute_ZIPln(endog, exog = exog, exog_inflation = exog, dict_initialization=dict_init, use_closed_form_prob = True)
# brute.fit(tol=0, nb_max_iteration=200, verbose=True)
# brute.show()
# print('log like zi', zi.loglike)
# print('log like brute', brute.loglike)
