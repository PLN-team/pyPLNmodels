import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyPLNmodels import (
    ZIPln,
    load_microcosm,
    get_zipln_simulated_count_data,
    load_model,
)
from pyPLNmodels.models import Brute_ZIPln

endog, exog = load_microcosm(n_samples=2000, dim=2000)
# endog, exog = load_microcosm()
exog_inflation = exog
offsets = None
# endog, exog, exog_inflation, offsets = get_zipln_simulated_count_data()

# zibrute = Brute_ZIPln(
#     endog,
#     exog=exog,
#     exog_inflation=exog_inflation,
#     offsets=offsets,
#     add_const_inflation=True,
#     add_const=True,
#     use_closed_form_prob=True,
# )
# zibrute.fit(nb_max_iteration=2000,tol = 0)
# zibrute.save("zibrute")
# init = load_model("zibrute")

ENH_CLOSED_KEY = "Enhanced Analytic"
ENH_FREE_KEY = "Enhanced"
STD_CLOSED_KEY = "Standard Analytic"
STD_FREE_KEY = "Standard"

models = {
    ENH_FREE_KEY: ZIPln(
        endog,
        exog=exog,
        offsets=offsets,
        add_const_inflation=True,
        exog_inflation=exog_inflation,
        add_const=True,
        use_closed_form_prob=False,
    ),
    ENH_CLOSED_KEY: ZIPln(
        endog,
        exog=exog,
        offsets=offsets,
        add_const_inflation=True,
        exog_inflation=exog_inflation,
        add_const=True,
        use_closed_form_prob=True,
        # dict_initialization=init,
    ),
    STD_FREE_KEY: Brute_ZIPln(
        endog,
        exog=exog,
        exog_inflation=exog_inflation,
        offsets=offsets,
        add_const_inflation=True,
        add_const=True,
        use_closed_form_prob=False,
    ),
    STD_CLOSED_KEY: Brute_ZIPln(
        endog,
        exog=exog,
        exog_inflation=exog_inflation,
        offsets=offsets,
        add_const_inflation=True,
        add_const=True,
        use_closed_form_prob=True,
    ),
}
dict_res = {}
for key, model in models.items():
    model.fit(nb_max_iteration=2000, tol=0, lr=0.001)
    # model.fit(nb_max_iteration=5, tol=0)
    y = model._criterion_args._elbos_list
    absc = np.arange(0, len(y))
    plt.plot(absc, -np.array(y), label=key)
    dict_res[key] = y

df = pd.DataFrame.from_dict(dict_res)
df.to_csv("elbos_res/dict_elbos_simu.csv")

plt.yscale("log")
plt.ylabel("ELBO")
plt.xlabel("Number of iterations")
plt.legend()
plt.show()


# plt.plot(absc, y, label=key)
# plt.legend()
# plt.show()
