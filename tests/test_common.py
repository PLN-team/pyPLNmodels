import os
import functools

import torch
import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture as lf

from pyPLNmodels.models import PLN, _PLNPCA
from tests.utils import MSE

# from tests.import_fixtures import get_dict_fixtures

from tests.conftest import dict_fixtures


# dict_fixtures_pln_full = dict_fixtures_models[0]
# dict_fixtures_plnpca = dict_fixtures_models[1]
# dict_fixturesplnpca = dict_fixtures_models[2]


def get_pln_and_plncpca_fixtures(key):
    return dict_fixtures_pln_full[key] + dict_fixtures_plnpca[key]


def get_pca_fixtures(key):
    return dict_fixtures_plnpca[key] + dict_fixturesplnpca[key]


def get_all_fixtures(key):
    return (
        dict_fixtures_plnpca[key]
        + dict_fixtures_pln_full[key]
        + dict_fixturesplnpca[key]
    )


def filter_models(models_name):
    def decorator(my_test):
        @functools.wraps(my_test)
        def new_test(**kwargs):
            fixture = next(iter(kwargs.values()))
            if type(fixture).__name__ not in models_name:
                return None
            return my_test(**kwargs)

        return new_test

    return decorator


# @pytest.mark.parametrize("any_pln", [dict_fixtures["simulated_pln_0cov_array"]])
# @pytest.mark.parametrize("any_pln", [dict_fixtures["simulated_pln_0cov_array"]])
@filter_models(["PLN", "_PLNPCA"])
def test_properties(simulated_fitted_pln_0cov_array):
    assert hasattr(simulated_fitted_pln_0cov_array, "model_parameters")
    assert hasattr(simulated_fitted_pln_0cov_array, "latent_parameters")
    print("model_param", simulated_fitted_pln_0cov_array.model_parameters)
    assert hasattr(simulated_fitted_pln_0cov_array, "latent_variables")
    assert hasattr(simulated_fitted_pln_0cov_array, "optim_parameters")


@pytest.mark.parametrize("any_pln", dict_fixtures["fitted_pln"])
def test_print(any_pln):
    print(any_pln)


print("len :", len(dict_fixtures["all_pln"]))

"""
@pytest.mark.parametrize("any_pln", all_fitted_models)
def test_show_coef_transform_covariance_pcaprojected(any_pln):
    any_pln.show()
    any_pln.plotargs.show_loss(savefig=True)
    any_pln.plotargs.show_stopping_criterion(savefig=True)
    assert hasattr(any_pln, "coef")
    assert callable(any_pln.transform)
    assert hasattr(any_pln, "covariance")
    assert callable(any_pln.pca_projected_latent_variables)
    assert any_pln.pca_projected_latent_variables(n_components=None) is not None
    with pytest.raises(Exception):
        any_pln.pca_projected_latent_variables(n_components=any_pln.dim + 1)


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


@pytest.mark.parametrize("any_instance_pln", all_instances)
def test_verbose(any_instance_pln):
    any_instance_pln.fit(
        counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim, verbose=True
    )


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_only_counts(sim_pln):
    sim_pln.fit(counts=counts_sim)


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_only_counts_and_offsets(sim_pln):
    sim_pln.fit(counts=counts_sim, offsets=offsets_sim)


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_only_Y_and_cov(sim_pln):
    sim_pln.fit(counts=counts_sim, covariates=covariates_sim)


@pytest.mark.parametrize("simulated_fitted_any_pln", simulated_any_pln)
def test_find_right_covariance(simulated_fitted_any_pln):
    mse_covariance = MSE(simulated_fitted_any_pln.covariance - true_covariance)
    assert mse_covariance < 0.05


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_find_right_coef(sim_pln):
    mse_coef = MSE(sim_pln.coef - true_coef)
    assert mse_coef < 0.1


def test_number_of_iterations_pln_full(simulated_fitted_pln_full_0cov):
    nb_iterations = len(simulated_fitted_pln_full_0cov.elbos_list)
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


def test_computable_elbo_full(instance_pln_full, simulated_fitted_pln_full_0cov):
    instance_pln_full.counts = simulated_fitted_pln_full_0cov.counts
    instance_pln_full.covariates = simulated_fitted_pln_full_0cov.covariates
    instance_pln_full.offsets = simulated_fitted_pln_full_0cov.offsets
    instance_pln_full.latent_mean = simulated_fitted_pln_full_0cov.latent_mean
    instance_pln_full.latent_var = simulated_fitted_pln_full_0cov.latent_var
    instance_pln_full.covariance = simulated_fitted_pln_full_0cov.covariance
    instance_pln_full.coef = simulated_fitted_pln_full_0cov.coef
    instance_pln_full.compute_elbo()


def test_fail_count_setter(simulated_fitted_pln_full_0cov):
    wrong_counts = torch.randint(size=(10, 5), low=0, high=10)
    with pytest.raises(Exception):
        simulated_fitted_pln_full_0cov.counts = wrong_counts


@pytest.mark.parametrize("any_pln", all_fitted_models)
def test_setter_with_numpy(any_pln):
    np_counts = any_pln.counts.numpy()
    any_pln.counts = np_counts


@pytest.mark.parametrize("any_pln", all_fitted_models)
def test_setter_with_pandas(any_pln):
    pd_counts = pd.DataFrame(any_pln.counts.numpy())
    any_pln.counts = pd_counts


@pytest.mark.parametrize("instance", all_instances)
def test_random_init(instance):
    instance.fit(counts_sim, covariates_sim, offsets_sim, do_smart_init=False)


@pytest.mark.parametrize("instance", all_instances)
def test_print_end_of_fitting_message(instance):
    instance.fit(counts_sim, covariates_sim, offsets_sim, nb_max_iteration=4)


@pytest.mark.parametrize("any_pln", all_fitted_models)
def test_fail_wrong_covariates_prediction(any_pln):
    X = torch.randn(any_pln.n_samples, any_pln.nb_cov)
    with pytest.raises(Exception):
        any_pln.predict(X)


@pytest.mark.parametrize("any__plnpca", all_fitted__plnpca)
def test_latent_var_pca(any__plnpca):
    assert any__plnpca.transform(project=False).shape == any__plnpca.counts.shape
    assert any__plnpca.transform().shape == (any__plnpca.n_samples, any__plnpca.rank)


@pytest.mark.parametrize("any_pln_full", all_fitted_pln_full)
def test_latent_var_pln_full(any_pln_full):
    assert any_pln_full.transform().shape == any_pln_full.counts.shape


def test_wrong_rank():
    instance = _PLNPCA(counts_sim.shape[1] + 1)
    with pytest.warns(UserWarning):
        instance.fit(counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim)


"""
