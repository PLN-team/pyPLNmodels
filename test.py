from pyPLNmodels import get_real_count_data, ZIPln, Pln, get_simulated_count_data
from pyPLNmodels.models import Brute_ZIPln
import matplotlib.pyplot as plt

endog, exog, offsets = get_simulated_count_data()

is_closed = False

brute_zi = Brute_ZIPln(
    endog, exog=exog, offsets=offsets, use_closed_form_prob=is_closed
)
brute_zi.fit()
print("brute elbo", brute_zi.elbo)
brute_zi.plot_expected_vs_true()


zi = ZIPln(endog, exog=exog, offsets=offsets, use_closed_form_prob=is_closed)
zi.fit()
print("elbo zi", zi.elbo)
zi.plot_expected_vs_true()
