import pytest
import torch

from pyPLNmodels import get_simulation_parameters, sample_pln, ZIPln
from tests.conftest import dict_fixtures
from tests.utils import filter_models, MSE


from pyPLNmodels import get_simulated_count_data


@pytest.mark.parametrize("zi", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_properties(zi):
    assert hasattr(zi, "latent_prob")
    assert hasattr(zi, "coef_inflation")


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_predict(model):
    X = torch.randn((model.n_samples, model.nb_cov))
    prediction = model.predict(X)
    expected = X @ model.coef
    assert torch.all(torch.eq(expected, prediction))


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_predict_prob(model):
    X = torch.randn((model.n_samples, model.nb_cov))
    prediction = model.predict_prob_inflation(X)
    expected = torch.sigmoid(X @ model.coef_inflation)
    assert torch.all(torch.eq(expected, prediction))


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_fail_predict_prob(model):
    X1 = torch.randn((model.n_samples, model.nb_cov + 1))
    X2 = torch.randn((model.n_samples, model.nb_cov - 1))
    with pytest.raises(RuntimeError):
        model.predict_prob_inflation(X1)
    with pytest.raises(RuntimeError):
        model.predict_prob_inflation(X2)


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_fail_predict(model):
    X1 = torch.randn((model.n_samples, model.nb_cov + 1))
    X2 = torch.randn((model.n_samples, model.nb_cov - 1))
    with pytest.raises(RuntimeError):
        model.predict(X1)
    with pytest.raises(RuntimeError):
        model.predict(X2)


@pytest.mark.parametrize("model", dict_fixtures["sim_model_0cov_fitted_and_loaded"])
@filter_models(["ZIPln"])
def test_no_exog_not_possible(model):
    assert model.nb_cov == 1
    assert model._coef_inflation.shape[0] == 1


def test_find_right_covariance_coef_and_infla():
    pln_param = get_simulation_parameters(zero_inflated=True, n_samples=1000)
    # pln_param._coef += 5
    endog = sample_pln(pln_param, seed=0, return_latent=False)
    exog = pln_param.exog
    offsets = pln_param.offsets
    covariance = pln_param.covariance
    coef = pln_param.coef
    coef_inflation = pln_param.coef_inflation
    endog, exog, offsets, covariance, coef, coef_inflation = get_simulated_count_data(
        zero_inflated=True, return_true_param=True, n_samples=1000
    )
    zi = ZIPln(endog, exog=exog, offsets=offsets, use_closed_form_prob=False)
    zi.fit()
    mse_covariance = MSE(zi.covariance.cpu() - covariance.cpu())
    mse_coef = MSE(zi.coef.cpu() - coef.cpu())
    mse_coef_infla = MSE(zi.coef_inflation.cpu() - coef_inflation.cpu())
    assert mse_coef < 3
    assert mse_coef_infla < 3
    assert mse_covariance < 1


@pytest.mark.parametrize("zi", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_latent_variables(zi):
    z, w = zi.latent_variables
    assert z.shape == zi.endog.shape
    assert w.shape == zi.endog.shape


@pytest.mark.parametrize("zi", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_transform(zi):
    z = zi.transform()
    assert z.shape == zi.endog.shape
    z, w = zi.transform(return_latent_prob=True)
    assert z.shape == w.shape == zi.endog.shape


def test_mse():
    n_samples = 300
    pln_param = get_simulation_parameters(
        zero_inflated=True, n_samples=n_samples, nb_cov=0
    )
    pln_param._coef += 6
    endog = sample_pln(pln_param, seed=0, return_latent=False)
    exog = pln_param.exog
    offsets = pln_param.offsets
    covariance = pln_param.covariance
    coef = pln_param.coef
    coef_inflation = pln_param.coef_inflation
    zi = ZIPln(endog, exog=exog, offsets=offsets, use_closed_form_prob=True)
    zi.fit()
    mse_covariance = MSE(zi.covariance.cpu() - covariance.cpu())
    mse_coef = MSE(zi.coef.cpu() - coef.cpu())
    mse_coef_infla = MSE(zi.coef_inflation.cpu() - coef_inflation.cpu())
    assert mse_coef < 0.05
    assert mse_coef_infla < 0.08
    assert mse_covariance < 0.3


@pytest.mark.parametrize("model", dict_fixtures["sim_model_instance"])
@filter_models(["ZIPln"])
def test_batch(model):
    model.batch_size = 20
    model.fit()
    print(model)
