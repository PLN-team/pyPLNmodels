from pyPLNmodels import ZIPln, load_microcosm
from pyPLNmodels.models import Brute_ZIPln

import matplotlib.pyplot as plt


ENH_CLOSED_KEY = "Enhanced Analytic"
ENH_FREE_KEY = "Enhanced"
STD_CLOSED_KEY = "Standard Analytic"
STD_FREE_KEY = "Standard"

endog, exog = load_microcosm()
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
    model.fit(nb_max_iteration=5)
