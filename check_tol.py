from pyPLNmodels import Pln, PlnPCA, get_pln_simulated_count_data, load_scrna
import matplotlib.pyplot as plt

n_samples = 300
nb_epoch = 50
batch_size = 10
endog, exog, offsets = get_pln_simulated_count_data(n_samples=n_samples, dim=100)

data = load_scrna()


# pln = Pln(endog, exog = exog, offsets = offsets, batch_size = batch_size)
pln = Pln.from_formula("endog ~ 1", data=data, batch_size=batch_size)
print("nb iter", nb_epoch * n_samples)
pln.fit(nb_max_iteration=int(nb_epoch * n_samples))

pln._all_mses
fig, axes = plt.subplots(2)
pln._display_norm(axes[0])
pln._display_ma(ma=4, ax=axes[1])
plt.show()
pln.show()
