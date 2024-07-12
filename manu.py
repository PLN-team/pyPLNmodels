import numpy as np
from pyPLNmodels import Pln, load_scrna


data = load_scrna()

pln = Pln(data["endog"])
pln.fit_vem(nb_gradient_steps=10, verbose=True)
print(pln)

pln = Pln(data["endog"])
pln.fit(verbose=True)
print(pln)
