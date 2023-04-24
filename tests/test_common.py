import torch
import numpy as np
from pyPLNmodels.models import PLN, _PLNPCA
from pyPLNmodels import get_simulated_count_data, get_real_count_data
from tests.utils import MSE

import pytest
from pytest_lazyfixture import lazy_fixture as lf
import os

os.chdir("./tests")
(
    counts_sim,
    covariates_sim,
    offsets_sim,
    true_covariance,
    true_coef,
) = get_simulated_count_data(return_true_param=True, nb_cov=2)


counts_real = get_real_count_data()
rank = 8


@pytest.fixture
def instance_pln_full():
    pln_full = PLN()
    return pln_full


@pytest.fixture
def instance__plnpca():
    plnpca = _PLNPCA(rank=rank)
    return plnpca


@pytest.fixture
def simulated_fitted_pln_full():
    pln_full = PLN()
    pln_full.fit(counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim)
    return pln_full


@pytest.fixture
def simulated_fitted__plnpca():
    plnpca = _PLNPCA(rank=rank)
    plnpca.fit(counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim)
    return plnpca


@pytest.fixture
def loaded_simulated_pln_full(simulated_fitted_pln_full):
    simulated_fitted_pln_full.save()
    loaded_pln_full = PLN()
    loaded_pln_full.load()
    return loaded_pln_full


@pytest.fixture
def loaded_refit_simulated_pln_full(loaded_simulated_pln_full):
    loaded_simulated_pln_full.fit(
        counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim
    )
    return loaded_simulated_pln_full


@pytest.fixture
def loaded_simulated__plnpca(simulated_fitted__plnpca):
    simulated_fitted__plnpca.save()
    loaded_pln_full = _PLNPCA(rank=rank)
    loaded_pln_full.load()
    return loaded_pln_full


@pytest.fixture
def loaded_refit_simulated__plnpca(loaded_simulated__plnpca):
    loaded_simulated__plnpca.fit(
        counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim
    )
    return loaded_simulated__plnpca


@pytest.fixture
def real_fitted_pln_full():
    pln_full = PLN()
    pln_full.fit(counts=counts_real)
    return pln_full


@pytest.fixture
def loaded_real_pln_full(real_fitted_pln_full):
    real_fitted_pln_full.save()
    loaded_pln_full = PLN()
    loaded_pln_full.load()
    return loaded_pln_full


@pytest.fixture
def loaded_refit_real_pln_full(loaded_real_pln_full):
    loaded_real_pln_full.fit(counts=counts_real)
    return loaded_real_pln_full


@pytest.fixture
def real_fitted__plnpca():
    plnpca = _PLNPCA(rank=rank)
    plnpca.fit(counts=counts_real)
    return plnpca


@pytest.fixture
def loaded_real__plnpca(real_fitted__plnpca):
    real_fitted__plnpca.save()
    loaded_plnpca = _PLNPCA(rank=rank)
    loaded_plnpca.load()
    return loaded_plnpca


@pytest.fixture
def loaded_refit_real__plnpca(loaded_real__plnpca):
    loaded_real__plnpca.fit(counts=counts_real)
    return loaded_real__plnpca


# all_fitted_models = [
#     lf("simulated_fitted_pln_full"),
#     lf("loaded_simulated_pln_full"),
#     lf("loaded_refit_simulated_pln_full"),
#     lf("simulated_fitted__plnpca"),
#     # lf("loaded_simulated__plnpca"),
#     # lf("loaded_refit_simulated__plnpca"),
#     # lf("real_fitted_pln_full"),
#     # lf("loaded_real_pln_full"),
#     # lf("loaded_refit_real_pln_full"),
#     # lf("real_fitted__plnpca"),
#     # lf("loaded_real__plnpca"),
#     # lf("loaded_refit_real__plnpca"),
# ]
real_pln_full = [
    lf("real_fitted_pln_full"),
    lf("loaded_real_pln_full"),
    lf("loaded_refit_real_pln_full"),
]
real__plnpca = [
    lf("real_fitted__plnpca"),
    lf("loaded_real__plnpca"),
    lf("loaded_refit_real__plnpca"),
]
simulated_pln_full = [
    lf("simulated_fitted_pln_full"),
    lf("loaded_simulated_pln_full"),
    lf("loaded_refit_simulated_pln_full"),
]
simulated__plnpca = [
    lf("simulated_fitted__plnpca"),
    lf("loaded_simulated__plnpca"),
    lf("loaded_refit_simulated__plnpca"),
]

all_fitted__plnpca = simulated__plnpca + real__plnpca
all_fitted_pln_full = simulated_pln_full + real_pln_full

simulated_any_pln = simulated__plnpca + simulated_pln_full
real_any_pln = real_pln_full + real__plnpca
all_fitted_models = simulated_any_pln + real_any_pln


@pytest.mark.parametrize("any_pln", all_fitted_models)
def test_properties(any_pln):
    assert hasattr(any_pln, "latent_variables")
    assert hasattr(any_pln, "model_parameters")
    assert hasattr(any_pln, "latent_parameters")
    assert hasattr(any_pln, "optim_parameters")


@pytest.mark.parametrize(
    "any_pln",
    all_fitted_models,
)
def test_show_coef_transform_covariance_pcaprojected(any_pln):
    outputs = []
    any_pln.show()
    assert hasattr(any_pln, "coef")
    assert callable(any_pln.transform)
    assert hasattr(any_pln, "covariance")
    assert callable(any_pln.pca_projected_latent_variables)
    assert any_pln.pca_projected_latent_variables(n_components=None) is not None


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_predict_simulated(sim_pln):
    X = torch.randn((sim_pln.n_samples, sim_pln.nb_cov - 1))
    prediction = sim_pln.predict(X)
    expected = (
        torch.stack((torch.ones(sim_pln.n_samples, 1), X), axis=1).squeeze()
        @ sim_pln.coef
    )
    assert torch.all(torch.eq(expected, prediction))


@pytest.mark.parametrize("real_pln", real_any_pln)
def test_predict_real(real_pln):
    prediction = real_pln.predict()
    expected = torch.ones(real_pln.n_samples, 1) @ real_pln.coef
    assert torch.all(torch.eq(expected, prediction))


@pytest.mark.parametrize("any_pln", all_fitted_models)
def test_print(any_pln):
    print(any_pln)


@pytest.mark.parametrize(
    "any_instance_pln", [lf("instance__plnpca"), lf("my_instance_pln_full")]
)
def test_verbose(any_instance_pln):
    any_instance_pln.fit(
        counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim, verbose=True
    )


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_only_Y(sim_pln):
    sim_pln.fit(counts=counts_sim)


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_only_Y_and_O(sim_pln):
    sim_pln.fit(counts=counts_sim, offsets=offsets_sim)


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_only_Y_and_cov(sim_pln):
    sim_pln.fit(counts=counts_sim, covariates=covariates_sim)


@pytest.mark.parametrize("plnpca", all_fitted__plnpca)
def test_loading_back_pca(plnpca):
    save_and_loadback_pca(plnpca)


@pytest.mark.parametrize("pln_full", all_fitted_pln_full)
def test_load_back_pln_full(pln_full):
    save_and_loadback_pca(pln_full)


@pytest.mark.parametrize("pln_full", all_fitted_pln_full)
def test_load_back_and_refit_pln_full(pln_full):
    save_and_loadback_pca(pln_full)
    pln_full.fit()


def save_and_loadback_pln_full(model):
    model.save()
    newpln_full = PLN()
    newpln_full.load()
    return newpln_full


def save_and_loadback_pca(plnpca):
    plnpca.save()
    new = _PLNPCA(rank=rank)
    new.load()
    return new


@pytest.mark.parametrize("simulated_fitted_any_pln", simulated_any_pln)
def test_find_right_covariance(simulated_fitted_any_pln):
    mse_covariance = MSE(simulated_fitted_any_pln.covariance - true_covariance)
    assert mse_covariance < 0.05


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_find_right_coef(sim_pln):
    mse_coef = MSE(sim_pln.coef - true_coef)
    assert mse_coef < 0.1


def test_number_of_iterations_pln_full(simulated_fitted_pln_full):
    nb_iterations = len(simulated_fitted_pln_full.elbos_list)
    assert 50 < nb_iterations < 300


def test_computable_elbopca(instance__plnpca, simulated_fitted__plnpca):
    instance__plnpca.counts = simulated_fitted__plnpca.counts
    instance__plnpca.covariates = simulated_fitted__plnpca.covariates
    instance__plnpca.offsets = simulated_fitted__plnpca.offsets
    instance__plnpca.latent_mean = simulated_fitted__plnpca.latent_mean
    instance__plnpca.latent_var = simulated_fitted__plnpca.latent_var
    instance__plnpca.components = simulated_fitted__plnpca.components
    instance__plnpca.coef = simulated_fitted__plnpca.coef
    instance__plnpca.compute_elbo()


def test_computable_elbo_full(instance_pln_full, simulated_fitted_pln_full):
    instance_pln_full.counts = simulated_fitted_pln_full.counts
    instance_pln_full.covariates = simulated_fitted_pln_full.covariates
    instance_pln_full.offsets = simulated_fitted_pln_full.offsets
    instance_pln_full.latent_mean = simulated_fitted_pln_full.latent_mean
    instance_pln_full.latent_var = simulated_fitted_pln_full.latent_var
    instance_pln_full.covariance = simulated_fitted_pln_full.covariance
    instance_pln_full.coef = simulated_fitted_pln_full.coef
    instance_pln_full.compute_elbo()
