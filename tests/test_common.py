import os

import torch
import pytest

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models

from tests.import_data import true_sim_0cov, true_sim_2cov, endog_real

pln_and_plnpca = ["Pln", "PlnPCA"]
single_models = ["Pln", "PlnPCA", "ZIPln"]


@pytest.mark.parametrize("any_model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(single_models)
def test_properties(any_model):
    assert hasattr(any_model, "latent_parameters")
    assert hasattr(any_model, "latent_variables")
    assert hasattr(any_model, "optim_parameters")
    assert hasattr(any_model, "model_parameters")


@pytest.mark.parametrize("sim_model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(pln_and_plnpca)
def test_predict_simulated(sim_model):
    if sim_model.nb_cov == 0:
        assert sim_model.predict() is None
        with pytest.raises(AttributeError):
            sim_model.predict(1)
    else:
        X = torch.randn((sim_model.n_samples, sim_model.nb_cov))
        prediction = sim_model.predict(X)
        expected = X @ sim_model.coef
        assert torch.all(torch.eq(expected, prediction))


@pytest.mark.parametrize("any_instance_model", dict_fixtures["instances"])
def test_verbose(any_instance_model):
    any_instance_model.fit(verbose=True, tol=0.1)


@pytest.mark.parametrize(
    "simulated_fitted_any_model", dict_fixtures["loaded_and_fitted_sim_model"]
)
@filter_models(pln_and_plnpca)
def test_find_right_covariance(simulated_fitted_any_model):
    if simulated_fitted_any_model.nb_cov == 0:
        true_covariance = true_sim_0cov["Sigma"].cpu()
    elif simulated_fitted_any_model.nb_cov == 2:
        true_covariance = true_sim_2cov["Sigma"].cpu()
    else:
        raise ValueError(
            f"Not the right numbers of covariance({simulated_fitted_any_model.nb_cov})"
        )
    mse_covariance = MSE(
        simulated_fitted_any_model.covariance.cpu() - true_covariance.cpu()
    )
    assert mse_covariance < 0.05


@pytest.mark.parametrize(
    "real_fitted_and_loaded_model", dict_fixtures["loaded_and_fitted_real_model"]
)
@filter_models(single_models)
def test_right_covariance_shape(real_fitted_and_loaded_model):
    assert real_fitted_and_loaded_model.covariance.shape == (
        endog_real.shape[1],
        endog_real.shape[1],
    )


@pytest.mark.parametrize(
    "simulated_fitted_any_model", dict_fixtures["loaded_and_fitted_model"]
)
@filter_models(pln_and_plnpca)
def test_find_right_coef(simulated_fitted_any_model):
    if simulated_fitted_any_model.nb_cov == 2:
        true_coef = true_sim_2cov["beta"]
        mse_coef = MSE(simulated_fitted_any_model.coef.cpu() - true_coef.cpu())
        assert mse_coef < 0.1
    elif simulated_fitted_any_model.nb_cov == 0:
        assert simulated_fitted_any_model.coef is None


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(single_models)
def test_fail_count_setter(model):
    wrong_endog = torch.randint(size=(10, 5), low=0, high=10)
    negative_endog = -model._endog
    with pytest.raises(ValueError):
        model.endog = wrong_endog
    with pytest.raises(ValueError):
        model.endog = negative_endog


@pytest.mark.parametrize("instance", dict_fixtures["instances"])
def test_random_init(instance):
    instance.fit(do_smart_init=False)


@pytest.mark.parametrize("instance", dict_fixtures["instances"])
def test__print_end_of_fitting_message(instance):
    instance.fit(nb_max_iteration=4)


@pytest.mark.parametrize("model", dict_fixtures["fitted_model"])
@filter_models(single_models)
def test_fail_wrong_exog_prediction(model):
    X = torch.randn(model.n_samples, model.nb_cov + 1)
    with pytest.raises(Exception):
        model.predict(X)


@pytest.mark.parametrize("model", dict_fixtures["sim_model_instance"])
@filter_models(pln_and_plnpca)
def test_batch(model):
    model.fit(batch_size=20)
    print(model)
    model.show()
    if model.nb_cov == 2:
        true_coef = true_sim_2cov["beta"]
        mse_coef = MSE(model.coef.cpu() - true_coef.cpu())
        assert mse_coef < 0.1
    elif model.nb_cov == 0:
        assert model.coef is None
