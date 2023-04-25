import pytest
from pytest_lazyfixture import lazy_fixture

from pyPLNmodels.VEM import PLN, PLNPCA
from tests.utils import get_simulated_data, MSE

RANKS = [2, 4]

Y_sim, covariates_sim, O_sim, true_Sigma, true_beta = get_simulated_data()


@pytest.fixture
def my_instance_plnpca():
    plnpca = PLNPCA(ranks=RANKS)
    return plnpca


@pytest.fixture
def simulated_fitted_plnpca():
    plnpca = PLNPCA(RANKS)
    plnpca.fit(Y_sim, covariates_sim, O_sim)
    return plnpca


def test_find_right_Sigma(simulated_fitted_plnpca):
    passed = True
    for model in simulated_fitted_plnpca.models:
        mse_Sigma = MSE(model.Sigma - true_Sigma)
        if mse_Sigma > 0.1:
            passed = False
    assert passed


def test_find_right_beta(simulated_fitted_plnpca):
    passed = True
    for model in simulated_fitted_plnpca.models:
        mse_beta = MSE(model.beta - true_beta)
        if mse_beta > 0.1:
            passed = False
    assert passed
