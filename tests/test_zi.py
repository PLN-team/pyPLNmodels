import pytest
import torch
import numpy as np

from pyPLNmodels import get_simulation_parameters, ZIPln, sample_zipln
from tests.conftest import dict_fixtures
from tests.utils import filter_models, MSE


from pyPLNmodels import get_zipln_simulated_count_data


def mae(t):
    return torch.mean(torch.abs(t))


def mse(t):
    return torch.mean(t**2)


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
    zipln_param = get_simulation_parameters(
        n_samples=1000, zero_inflation_formula="column-wise"
    )
    # pln_param._coef += 5
    endog = sample_zipln(zipln_param, seed=0, return_latent=False)
    exog = zipln_param.exog
    offsets = zipln_param.offsets
    covariance = zipln_param.covariance
    coef = zipln_param.coef
    coef_inflation = zipln_param.coef_inflation
    (
        endog,
        exog,
        exog_infla,
        offsets,
        covariance,
        coef,
        coef_inflation,
    ) = get_zipln_simulated_count_data(
        return_true_param=True, n_samples=1000, zero_inflation_formula="column-wise"
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
    zipln_param = get_simulation_parameters(
        zero_inflation_formula="column-wise", n_samples=n_samples, nb_cov=1
    )
    zipln_param._coef += 6
    endog = sample_zipln(zipln_param, seed=0, return_latent=False)
    exog = zipln_param.exog
    offsets = zipln_param.offsets
    covariance = zipln_param.covariance
    coef = zipln_param.coef
    coef_inflation = zipln_param.coef_inflation
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


def test_zi_columns():
    n_samples = 100
    zipln_param = get_simulation_parameters(
        zero_inflation_formula="column-wise", n_samples=n_samples, nb_cov_inflation=1
    )
    endog = sample_zipln(zipln_param)
    exog = zipln_param.exog
    exog_inflation = zipln_param.exog_inflation
    offsets = zipln_param.offsets
    zi = ZIPln(endog, exog=exog, exog_inflation=exog_inflation, offsets=offsets)


def train_zi(formula):
    n_samples = 100
    if formula == "global":
        nb_cov_inflation = 0
        add_const_inflation = False
    elif formula in ["column-wise", "row-wise"]:
        nb_cov_inflation = 1
        add_const_inflation = True
    zipln_param = get_simulation_parameters(
        zero_inflation_formula=formula,
        n_samples=n_samples,
        nb_cov_inflation=nb_cov_inflation,
        dim=50,
        add_const_inflation=add_const_inflation,
    )
    zipln_param._coef += 4
    endog = sample_zipln(zipln_param)
    exog = zipln_param.exog
    exog_inflation = zipln_param.exog_inflation
    offsets = zipln_param.offsets
    zi = ZIPln(
        endog,
        exog=exog,
        exog_inflation=exog_inflation,
        offsets=offsets,
        zero_inflation_formula=formula,
        add_const_inflation=False,
        use_closed_form_prob=False,
    )
    zi.fit()
    return zi, zipln_param


def test_ziglobal():
    zi, zipln_param = train_zi("global")
    assert mae(zi.coef_inflation - zipln_param.coef_inflation) < 0.15
    assert mae(zi.proba_inflation - zipln_param.proba_inflation) < 0.03
    assert mae(zi.coef - zipln_param.coef) < 0.1
    assert mae(zi.covariance - zipln_param.covariance) < 0.3


def test_zicolumn():
    zi, zipln_param = train_zi("column-wise")
    assert mae(zi.coef_inflation - zipln_param.coef_inflation) < 0.3
    assert mae(zi.proba_inflation - zipln_param.proba_inflation) < 0.06
    assert mae(zi.coef - zipln_param.coef) < 0.4
    assert mae(zi.covariance - zipln_param.covariance) < 0.4


def test_zirow():
    zi, zipln_param = train_zi("row-wise")
    assert mae(zi.proba_inflation - zipln_param.proba_inflation) < 0.15
    print("coefinfla", zipln_param.coef_inflation)
    print("mine", zi._coef_inflation)
    x
    assert mae(zi.coef - zipln_param.coef) < 0.3
    assert mae(zi.covariance - zipln_param.covariance) < 0.3
