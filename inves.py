from pyPLNmodels import (
    Pln,
    ZIPln,
    load_microcosm,
    load_model,
    get_zipln_simulated_count_data,
)
from pyPLNmodels.models import Brute_ZIPln
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# dim = 97 -> 98 pour le standard sans forme close pour la proba (i.e. toutes les formes closes possibles)
# 61 pour le standard avec global
endog, exog = load_microcosm(dim=200, n_samples=500)
exog_inflation = exog
offsets = None
# endog, exog, exog_inflation, offsets = get_zipln_simulated_count_data()
nb_iter = 1000

dict_init = None


def show_model(model):
    # dict_init = load_model("zi")
    # mod = model(endog, exog = exog, exog_inflation = exog_inflation, offsets = offsets, dict_initialization = dict_init, use_closed_form_prob = True)
    mod = model(
        endog,
        exog=None,
        exog_inflation=None,
        offsets=offsets,
        dict_initialization=dict_init,
        use_closed_form_prob=True,
    )
    # mod = model(endog,zero_inflation_formula = "global", offsets = offsets, dict_initialization = dict_init, use_closed_form_prob = True)
    mod.fit(verbose=True, lr=0.001, tol=0, nb_max_iteration=nb_iter)
    not_diagonal = (mod._covariance * (1 - torch.eye(mod.dim))).detach()
    # sns.heatmap(not_diagonal)
    # plt.show()
    print("mean not diagonal", torch.mean((not_diagonal) ** 2))
    mod.show()
    mod.save("zi")


show_model(ZIPln)
# show_model(Brute_ZIPln)
# zi.save("zimodel")


# endog, exog, offsets = get_zipln_simulated_count_data()
