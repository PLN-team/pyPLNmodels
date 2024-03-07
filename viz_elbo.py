import matplotlib.pyplot as plt
import numpy as np

from pyPLNmodels import ZIPln, load_microcosm
from pyPLNmodels.models import Brute_ZIPln

# endog, exog = load_microcosm(n_samples = 2000, dim = 2000)
endog, exog = load_microcosm()

ENH_CLOSED_KEY = "Enhanced Analytic"
ENH_FREE_KEY = "Enhanced"
STD_CLOSED_KEY = "Standard Analytic"
STD_FREE_KEY = "Standard"

models = {
    ENH_FREE_KEY: ZIPln(
        endog,
        exog=exog,
        add_const_inflation=True,
        add_const=True,
        use_closed_form_prob=False,
    ),
    ENH_CLOSED_KEY: ZIPln(
        endog,
        exog=exog,
        add_const_inflation=True,
        add_const=True,
        use_closed_form_prob=True,
    ),
    STD_FREE_KEY: Brute_ZIPln(
        endog,
        exog=exog,
        add_const_inflation=True,
        add_const=True,
        use_closed_form_prob=False,
    ),
    STD_CLOSED_KEY: Brute_ZIPln(
        endog,
        exog=exog,
        add_const_inflation=True,
        add_const=True,
        use_closed_form_prob=True,
    ),
}

for key, model in models.items():
    model.fit(nb_max_iteration=1000, tol=0)
    y = model._criterion_args._elbos_list
    absc = np.arange(0, len(y))
    plt.plot(absc, y, label=key)
plt.legend()
plt.yscale("log")
plt.show()
