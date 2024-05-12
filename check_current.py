from pyPLNmodels import Pln, PlnPCA, get_pln_simulated_count_data, load_scrna, ZIPln
import matplotlib.pyplot as plt
import numpy as np

n_samples = 200
nb_epoch = 1500
endog, exog, offsets = get_pln_simulated_count_data(n_samples=n_samples, dim=100)

data = load_scrna(return_labels=True)


pln = ZIPln.from_formula("endog ~ 1", data=data)
pln.fit()
pln.viz(colors=data["labels"])
pln.show()

pln = ZIPln.from_formula("endog ~ 1", data=data)
pln.fit(tol=1e-1)
pln.viz(colors=data["labels"])
pln.show()

# pln = PlnPCA.from_formula("endog ~ 1", data=data, rank = 5)
# pln.fit(tol = 1e-7)
# pln.viz(colors = data["labels"])
# pln.show()
