
# PLNmodels: Poisson lognormal models

> The Poisson lognormal model and variants can be used for analysis of mutivariate count data.
> This package implements
> efficient algorithms to fit such models.
<!-- accompanied with a set of -->
<!-- > functions for visualization and diagnostic. See [this deck of -->
<!-- > slides](https://pln-team.github.io/slideshow/) for a -->
<!-- > comprehensive introduction. -->

## Getting started
The getting started can be found [here](https://forgemia.inra.fr/bbatardiere/pyplnmodels/-/raw/dev/Getting_started.ipynb?inline=false). If you need just a quick view of the package, see next.

## Installation

**PLNmodels** is available on
[pypi](https://pypi.org/project/pyPLNmodels/). The development
version is available on [GitHub](https://github.com/PLN-team/pyPLNmodels).

### Package installation

```
pip install pyPLNmodels
```

## Usage and main fitting functions

The package comes with an ecological data set to present the functionality
```
import pyPLNmodels
from pyPLNmodels.models import PlnPCAcollection, Pln
from pyPLNmodels.oaks import load_oaks
oaks = load_oaks()
```

### Unpenalized Poisson lognormal model (aka PLN)

```
pln = Pln.from_formula("counts ~ 1  + tree + dist2ground + orientation ", data = oaks, take_log_offsets = True)
pln.fit()
print(pln)
```


### Rank Constrained Poisson lognormal for Poisson Principal Component Analysis (aka PLNPCA)

```
pca =  PlnPCAcollection.from_formula("counts ~ 1  + tree + dist2ground + orientation ", data = oaks, take_log_offsets = True, ranks = [3,4,5])
pca.fit()
print(pca)
```


## References

Please cite our work using the following references:
-   J. Chiquet, M. Mariadassou and S. Robin: Variational inference for
    probabilistic Poisson PCA, the Annals of Applied Statistics, 12:
        2674–2698, 2018. [link](http://dx.doi.org/10.1214/18%2DAOAS1177)
