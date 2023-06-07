import os

import torch
import pytest

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models

from tests.import_data import true_sim_0cov, true_sim_2cov, counts_real


@pytest.mark.parametrize("any_pln", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_properties(any_pln):
    assert hasattr(any_pln, "latent_parameters")
    assert hasattr(any_pln, "latent_variables")
    assert hasattr(any_pln, "optim_parameters")
    assert hasattr(any_pln, "model_parameters")


@pytest.mark.parametrize("sim_pln", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_predict_simulated(sim_pln):
    if sim_pln.nb_cov == 0:
        assert sim_pln.predict() is None
        with pytest.raises(AttributeError):
            sim_pln.predict(1)
    else:
        X = torch.randn((sim_pln.n_samples, sim_pln.nb_cov))
        prediction = sim_pln.predict(X)
        expected = X @ sim_pln.coef
        assert torch.all(torch.eq(expected, prediction))


@pytest.mark.parametrize("any_instance_pln", dict_fixtures["instances"])
def test_verbose(any_instance_pln):
    any_instance_pln.fit(verbose=True, tol=0.1)


@pytest.mark.parametrize(
    "simulated_fitted_any_pln", dict_fixtures["loaded_and_fitted_sim_pln"]
)
@filter_models(["Pln", "PlnPCA"])
def test_find_right_covariance(simulated_fitted_any_pln):
    if simulated_fitted_any_pln.nb_cov == 0:
        true_covariance = true_sim_0cov["Sigma"]
    elif simulated_fitted_any_pln.nb_cov == 2:
        true_covariance = true_sim_2cov["Sigma"]
    mse_covariance = MSE(simulated_fitted_any_pln.covariance - true_covariance)
    assert mse_covariance < 0.05


@pytest.mark.parametrize(
    "real_fitted_and_loaded_pln", dict_fixtures["loaded_and_fitted_real_pln"]
)
@filter_models(["Pln", "PlnPCA"])
def test_right_covariance_shape(real_fitted_and_loaded_pln):
    assert real_fitted_and_loaded_pln.covariance.shape == (
        counts_real.shape[1],
        counts_real.shape[1],
    )


@pytest.mark.parametrize(
    "simulated_fitted_any_pln", dict_fixtures["loaded_and_fitted_pln"]
)
@filter_models(["Pln", "PlnPCA"])
def test_find_right_coef(simulated_fitted_any_pln):
    if simulated_fitted_any_pln.nb_cov == 2:
        true_coef = true_sim_2cov["beta"]
        mse_coef = MSE(simulated_fitted_any_pln.coef - true_coef)
        assert mse_coef < 0.1
    elif simulated_fitted_any_pln.nb_cov == 0:
        assert simulated_fitted_any_pln.coef is None


@pytest.mark.parametrize("pln", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_fail_count_setter(pln):
    wrong_counts = torch.randint(size=(10, 5), low=0, high=10)
    with pytest.raises(Exception):
        pln.counts = wrong_counts


@pytest.mark.parametrize("instance", dict_fixtures["instances"])
def test_random_init(instance):
    instance.fit(do_smart_init=False)


@pytest.mark.parametrize("instance", dict_fixtures["instances"])
def test__print_end_of_fitting_message(instance):
    instance.fit(nb_max_iteration=4)


@pytest.mark.parametrize("pln", dict_fixtures["fitted_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_fail_wrong_covariates_prediction(pln):
    X = torch.randn(pln.n_samples, pln.nb_cov + 1)
    with pytest.raises(Exception):
        pln.predict(X)
