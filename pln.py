import pandas as pd
import numpy as np
import torch

from pyPLNmodels import Pln, ZIPln, Brute_ZIPln


data = {}
data["endog"] = pd.read_csv("reduced/endog.csv").drop(columns="Unnamed: 0")
p = data["endog"].shape[1]
data["site"] = pd.read_csv("reduced/site.csv").drop(columns="Unnamed: 0").squeeze()
data["time"] = pd.read_csv("reduced/time.csv").drop(columns="Unnamed: 0").squeeze()


best = (data["endog"] > 0).mean(axis=0) > 0.05
data["endog"] = data["endog"].loc[:, best]


print("data shape", data["endog"].shape)

data["offsets"] = np.log(
    pd.read_csv("reduced/offsets.csv")
    .drop(columns="Unnamed: 0")
    .values.repeat(p, axis=1)
)[:, best]

zi = Brute_ZIPln.from_formula(
    "endog ~ 1", data=data, zero_inflation_formula="global", use_closed_form_prob=False
)
zi.fit(verbose=True)
print(zi)
print(zi.proba_inflation)


# pln = Pln.from_formula("endog ~ 1", data=data)
# pln.fit()
# print(pln)
