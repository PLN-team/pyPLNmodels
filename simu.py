from pyPLNmodels import get_real_count_data, ZIPln, Pln, get_simulated_count_data
from pyPLNmodels.models import Brute_ZIPln
import matplotlib.pyplot as plt

ENH_CLOSED_KEY = "enhanced_closed"
ENH_FREE_KEY = "enhanced_free"
STD_CLOSED_KEY = "standard_closed"
STD_FREE_KEY = "standard_free"


def get_dict_model(endog, exog, offsets):
    sim_models = {
        ENH_FREE_KEY: ZIPln(endog, exog=exog, offsets=offsets),
        ENH_CLOSED_KEY: ZIPln(endog, exog=exog, offsets=offsets, use_closed_form=True),
        STD_FREE_KEY: Brute_ZIPln(endog, exog=exog, offsets=offsets),
        STD_CLOSED_KEY: Brute_ZIPln(
            endog, exog=exog, offsets=offsets, use_closed_form=True
        ),
    }
    return sim_models
