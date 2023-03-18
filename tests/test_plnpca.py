import pytest
from pytest_lazyfixture import lazy_fixture

from pyPLNmodels.VEM import PLN, PLNPCA


RANKS = [2, 4]


@pytest.fixture
def my_instance_plnpca():
    plnpca = PLNPCA(ranks=RANKS)
    return plnpca


@pytest.fixture
def my_simulated_fitted_plnpca():
    plnpca = PLNPCA(RANKS)
    plnpca.fit(Y, covariates, O)
    return plnpca
