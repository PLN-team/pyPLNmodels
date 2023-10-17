import pytest
import torch

from pyPLNmodels import get_simulation_parameters, sample_pln, ZIPln
from tests.conftest import dict_fixtures
from tests.utils import filter_models, MSE


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


def test_find_right_covariance_and_coef():
    pln_param = get_simulation_parameters(
        n_samples=300, dim=50, nb_cov=2, rank=5, add_const=True
    )
    pln_param._coef += 5
    endog = sample_pln(pln_param, seed=0, return_latent=False)
    zi = ZIPln(endog, exog=pln_param.exog, offsets=pln_param.offsets)
    zi.fit()
    mse_covariance = MSE(zi.covariance - pln_param.covariance)
    mse_coef = MSE(zi.coef)
    assert mse_covariance < 0.5
