
# PLNmodels: Poisson lognormal models

> The Poisson lognormal model and variants can be used for analysis of mutivariate count data.
> This package implements
> efficient algorithms extracting meaningful data from difficult to interpret
> and complex multivariate count data. It has been built to scale on large datasets even
> though it has memory limitations. Possible fields of applications include
> - Genomics (number of times a gene is expressed in a cell)
> - Ecology (species abundances)
>
> One main functionality is to normalize the count data to obtain more valuable
> data. It also analyse the significance of each variable and their correlation as well as the weight of
> covariates (if available).
<!-- accompanied with a set of -->
<!-- > functions for visualization and diagnostic. See [this deck of -->
<!-- > slides](https://pln-team.github.io/slideshow/) for a -->
<!-- > comprehensive introduction. -->

##  Getting started
The getting started can be found [here](Getting_started.ipynb). If you need just a quick view of the package, see the quickstart next.

## 🛠 Installation

**pyPLNmodels** is available on
[pypi](https://pypi.org/project/pyPLNmodels/). The development
version is available on [GitHub](https://github.com/PLN-team/pyPLNmodels).

### Package installation

```
pip install pyPLNmodels
```

## ⚡️ Quickstart

The package comes with an ecological data set to present the functionality
```
import pyPLNmodels
from pyPLNmodels.models import PlnPCAcollection, Pln, ZIPln
from pyPLNmodels.oaks import load_oaks
oaks = load_oaks()
```

### Unpenalized Poisson lognormal model (aka PLN)

```
pln = Pln.from_formula("endog ~ 1  + tree + dist2ground + orientation ", data = oaks, take_log_offsets = True)
pln.fit()
print(pln)
transformed_data = pln.transform()
```


### Rank Constrained Poisson lognormal for Poisson Principal Component Analysis (aka PLNPCA)

```
pca =  PlnPCAcollection.from_formula("endog ~ 1  + tree + dist2ground + orientation ", data = oaks, take_log_offsets = True, ranks = [3,4,5])
pca.fit()
print(pca)
transformed_data = pca.best_model().transform()
```
### Zero inflated Poisson Log normal Model (aka ZIPln)
```
zi =  ZIPln.from_formula("endog ~ 1  + tree + dist2ground + orientation ", data = oaks, take_log_offsets = True)
zi.fit()
print(zi)
transformed_data = zi.transform()
```


## 👐 Contributing

Feel free to contribute, but read the [CONTRIBUTING.md](https://forgemia.inra.fr/bbatardiere/pyplnmodels/-/blob/main/CONTRIBUTING.md) first. A public roadmap will be available soon.


## ⚡️ Citations

Please cite our work using the following references:
-   J. Chiquet, M. Mariadassou and S. Robin: Variational inference for
    probabilistic Poisson PCA, the Annals of Applied Statistics, 12:
        2674–2698, 2018. [link](http://dx.doi.org/10.1214/18%2DAOAS1177)
