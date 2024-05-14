from pyPLNmodels import Pln, PlnPCA, get_pln_simulated_count_data, load_scrna, ZIPln
import matplotlib.pyplot as plt
import numpy as np

n_samples = 200
batch_size = 10
nb_epoch = 300
endog, exog, offsets = get_pln_simulated_count_data(n_samples=n_samples, dim=100)

data = load_scrna(return_labels=True)

pln = PlnPCA.from_formula("endog ~ 1", data=data)
# pln = PlnPCA(endog, exog = exog, offsets = offsets)
pln.fit(nb_max_iteration=nb_epoch)
pln.show()
# print('elbo ', pln.compute_elbo())
pln_batch = PlnPCA.from_formula("endog ~ 1", data=data, batch_size=batch_size)
# pln_batch = PlnPCA(endog, exog = exog, offsets = offsets, batch_size = batch_size)
pln_batch.fit(nb_max_iteration=nb_epoch, lr=0.001)
print("elbo batch", pln_batch.compute_elbo())

# pln = PlnPCA.from_formula("endog ~ 1", data=data, rank = 5)
# pln.fit(tol = 1e-7)
# pln.viz(colors = data["labels"])
# pln.show()
