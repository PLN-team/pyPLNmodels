import torch
import numpy as np
from pyPLNmodels.VEM import PLN, _PLNPCA
from pyPLNmodels import get_simulated_count_data, get_real_count_data
from tests.utils import MSE

import pytest
from pytest_lazyfixture import lazy_fixture as lf
import os

(
    counts_sim,
    covariates_sim,
    offsets_sim,
    true_Sigma,
    true_beta,
) = get_simulated_count_data(return_true_param=True)


counts_real = get_real_count_data()
rank = 8


@pytest.fixture
def my_instance_pln():
    pln = PLN()
    return pln


@pytest.fixture
def my_instance__plnpca():
    plnpca = _PLNPCA(rank=rank)
    return plnpca


@pytest.fixture
def my_simulated_fitted_pln():
    pln = PLN()
    pln.fit(counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim)
    return pln


@pytest.fixture
def my_real_fitted_pln():
    pln = PLN()
    pln.fit(counts=counts_real)
    return pln


@pytest.fixture
def my_simulated_fitted__plnpca():
    plnpca = _PLNPCA(rank=rank)
    plnpca.fit(counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim)
    return plnpca


@pytest.fixture
def my_real_fitted__plnpca():
    plnpca = _PLNPCA(rank=rank)
    plnpca.fit(counts=counts_real)
    return plnpca


@pytest.mark.parametrize(
    "simulated_fitted_any_pln",
    [lf("my_simulated_fitted_pln"), lf("my_simulated_fitted__plnpca")],
)
def test_find_right_Sigma(simulated_fitted_any_pln):
    mse_Sigma = MSE(simulated_fitted_any_pln.Sigma - true_Sigma)
    assert mse_Sigma < 0.05


@pytest.mark.parametrize(
    "pln", [lf("my_simulated_fitted_pln"), lf("my_simulated_fitted__plnpca")]
)
def test_find_right_beta(pln):
    mse_beta = MSE(pln.beta - true_beta)
    assert mse_beta < 0.1


def test_number_of_iterations(my_simulated_fitted_pln):
    nb_iterations = len(my_simulated_fitted_pln.elbos_list)
    assert 50 < nb_iterations < 150


@pytest.mark.parametrize(
    "any_pln",
    [
        lf("my_simulated_fitted_pln"),
        lf("my_simulated_fitted__plnpca"),
        lf("my_real_fitted_pln"),
        lf("my_real_fitted__plnpca"),
    ],
)
def test_properties(any_pln):
    latent_var = any_pln.latent_variables
    model_param = any_pln.model_parameters
    var_param = any_pln.var_parameters
    optim_param = any_pln.optim_parameters


@pytest.mark.parametrize(
    "any_pln",
    [
        lf("my_simulated_fitted_pln"),
        lf("my_simulated_fitted__plnpca"),
        lf("my_real_fitted_pln"),
        lf("my_real_fitted__plnpca"),
    ],
)
def test_show_coef_transform_sigma_pcaprojected(any_pln):
    outputs = []
    any_pln.show()
    outputs.append(any_pln.coef())
    outputs.append(any_pln.transform())
    outputs.append(any_pln.sigma())
    outputs.append(any_pln.pca_projected_latent_variables())
    outputs.append(any_pln.pca_projected_latent_variables(n_components=2))
    for output in outputs:
        if (isinstance(output, torch.Tensor)) is False:
            return False
    return True


@pytest.mark.parametrize(
    "sim_pln",
    [
        lf("my_simulated_fitted_pln"),
        lf("my_simulated_fitted__plnpca"),
    ],
)
def test_predict(sim_pln):
    X = torch.randn((sim_pln.n, sim_pln.d - 1))
    prediction = sim_pln.predict(X)
    expected = (
        torch.stack((torch.ones(sim_pln._n, 1), X), axis=1).squeeze() @ sim_pln.beta
    )
    assert torch.all(torch.eq(expected, prediction))


@pytest.mark.parametrize(
    "any_pln",
    [
        lf("my_simulated_fitted_pln"),
        lf("my_simulated_fitted__plnpca"),
        lf("my_real_fitted_pln"),
        lf("my_real_fitted__plnpca"),
    ],
)
def test_print(any_pln):
    print(any_pln)


@pytest.mark.parametrize(
    "any_instance_pln", [lf("my_instance__plnpca"), lf("my_instance_pln")]
)
def test_verbose(any_instance_pln):
    any_instance_pln.fit(
        counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim, verbose=True
    )


@pytest.mark.parametrize(
    "any_pln", [lf("my_simulated_fitted_pln"), lf("my_simulated_fitted__plnpca")]
)
def test_only_Y(any_pln):
    any_pln.fit(counts=counts_sim)


@pytest.mark.parametrize(
    "any_pln", [lf("my_simulated_fitted_pln"), lf("my_simulated_fitted__plnpca")]
)
def test_only_Y_and_O(any_pln):
    any_pln.fit(counts=counts_sim, offsets=offsets_sim)


@pytest.mark.parametrize(
    "any_pln", [lf("my_simulated_fitted_pln"), lf("my_simulated_fitted__plnpca")]
)
def test_only_Y_and_cov(any_pln):
    any_pln.fit(counts=counts_sim, covariates=covariates_sim)
