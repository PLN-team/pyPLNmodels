import pandas as pd
import os
import numpy as np
import torch
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

data = load_microcosm(get_interaction=False, n_samples=200, dim=120, for_formula=True)
# endog, exog, exog_inflation, offsets  = get_zipln_simulated_count_data(zero_inflation_formula = "column-wise", nb_cov_inflation = 3)

dict_init = None
# zi = ZIPln.from_formula("endog ~ 1 + exog", data=data, dict_initialization=dict_init)
zi = ZIPln(
    data["endog"],
    exog=data["exog"],
    exog_inflation=data["exog"],
    dict_initialization=dict_init,
)
# zi = ZIPln(endog, exog = exog, exog_inflation = exog_inflation, dict_initialization=dict_init)
zi.fit(tol=0, nb_max_iteration=300, verbose=True)
# zi.show()
