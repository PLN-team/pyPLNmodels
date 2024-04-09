import pandas as pd
import numpy as np

from pyPLNmodels import Pln


data = {}
data["endog"] = pd.read_csv("reduced/endog.csv").drop(columns="Unnamed: 0")
p = data["endog"].shape[1]
data["site"] = pd.read_csv("reduced/site.csv").drop(columns="Unnamed: 0").squeeze()
data["time"] = pd.read_csv("reduced/time.csv").drop(columns="Unnamed: 0").squeeze()


best = (data["endog"] > 0).mean(axis=0) > 0.05
data["endog"] = data["endog"].loc[:, best]

data["offsets"] = np.log(
    pd.read_csv("reduced/offsets.csv")
    .drop(columns="Unnamed: 0")
    .values.repeat(p, axis=1)
)[:, best]


pln = Pln.from_formula("endog ~ 1", data=data)
pln.fit()
print(pln)
