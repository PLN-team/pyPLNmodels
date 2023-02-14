import pandas as pd
import torch
from pyPLNmodels.VEM import ZIPLN, PLN, PLNPCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytest
from pytest_lazyfixture import lazy_fixture

Y = pd.read_csv("./example_data/test_data/Y_test.csv")
covariates = pd.read_csv("./example_data/test_data/cov_test.csv")
O = pd.read_csv("./example_data/test_data/O_test.csv")
true_Sigma = torch.from_numpy(
    pd.read_csv("./example_data/test_data/true_parameters/true_Sigma_test.csv").values
)
true_beta = torch.from_numpy(
    pd.read_csv("./example_data/test_data/true_parameters/true_beta_test.csv").values
)

# pln = PLN()
# pln.fit(Y, covariates, O)


def MSE(t):
    return torch.mean(t**2)


@pytest.fixture
def my_instance_pln():
    pln = PLN()
    return pln


@pytest.fixture
def my_instance_plnpca():
    plnpca = PLNPCA(q=8)
    return plnpca


@pytest.fixture
def my_test_pln():
    pln = PLN()
    pln.fit(Y, covariates, O)
    return pln


@pytest.fixture
def my_test_plnpca():
    plnpca = PLNPCA(q=10)
    plnpca.fit(Y, covariates, O)
    return plnpca


@pytest.mark.parametrize(
    "pln", [lazy_fixture("my_test_pln"), lazy_fixture("my_test_plnpca")]
)
def test_find_right_Sigma(pln):
    mse_Sigma = MSE(pln.get_Sigma() - true_Sigma)
    assert mse_Sigma < 0.01


@pytest.mark.parametrize(
    "pln", [lazy_fixture("my_test_pln"), lazy_fixture("my_test_plnpca")]
)
def test_find_right_beta(pln):
    mse_beta = MSE(pln.get_beta() - true_beta)
    assert mse_beta < 0.01


def test_number_of_iterations(my_test_pln):
    nb_iterations = len(my_test_pln.ELBOs_list)
    assert 40 < nb_iterations < 60


def test_plot(my_test_pln):
    my_test_pln.show()


@pytest.mark.parametrize(
    "pln", [lazy_fixture("my_test_pln"), lazy_fixture("my_test_plnpca")]
)
def test_verbose(pln):
    pln.fit(Y, covariates, O, verbose=True)


@pytest.mark.parametrize(
    "pln", [lazy_fixture("my_test_pln"), lazy_fixture("my_test_plnpca")]
)
def test_only_Y(pln):
    pln.fit(Y)


@pytest.mark.parametrize(
    "pln", [lazy_fixture("my_test_pln"), lazy_fixture("my_test_plnpca")]
)
def test_only_Y_and_O(pln):
    pln.fit(Y, O)


@pytest.mark.parametrize(
    "pln", [lazy_fixture("my_test_pln"), lazy_fixture("my_test_plnpca")]
)
def test_only_Y_and_cov(pln):
    pln.fit(Y, covariates)
