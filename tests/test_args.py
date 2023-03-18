from pyPLNmodels.VEM import PLN, PLNPCA
import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np
from tests.utils import get_simulated_data, get_real_data, MSE

Y_sim, covariates_sim, O_sim, true_Sigma, true_beta = get_simulated_data()

RANKS = [4, 8]


@pytest.fixture
def my_instance_plnpca():
    plnpca = PLNPCA(ranks=RANKS)
    return plnpca


def test_pandas_init(my_instance_plnpca):
    my_instance_plnpca.fit(Y_sim, covariates_sim, O_sim)
