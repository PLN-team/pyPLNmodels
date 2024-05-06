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


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_predict_and_properties_latent_variables(model):
    if model.nb_cov > 0:
        X = torch.randn((model.n_samples, model.nb_cov))
        prediction = model.predict(X)
        expected = X @ model.coef
        assert torch.all(torch.eq(expected, prediction))
    else:
        with pytest.raises(AttributeError):
            model.predict(1)
    assert hasattr(model, "latent_prob")
    assert hasattr(model, "coef_inflation")
    z, w = model.latent_variables
    assert z.shape == model.endog.shape
    assert w.shape == model.endog.shape


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_predictprob(model):
    if model._zero_inflation_formula != "global":
        if model._zero_inflation_formula == "column-wise":
            X = torch.randn((10, model.nb_cov_inflation))
            prediction = model.predict_prob_inflation(exog_infla=X)
            expected = torch.sigmoid(X @ model.coef_inflation)
        else:
            X = torch.randn((model.nb_cov_inflation, model.dim))
            prediction = model.predict_prob_inflation(exog_infla=X)
            expected = torch.sigmoid(model.coef_inflation @ X)
        assert torch.all(torch.eq(expected, prediction))
    else:
        prediction = model.predict_prob_inflation()
        assert prediction == torch.sigmoid(model.coef_inflation)


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_fail_predict_prob(model):
    if model._zero_inflation_formula != "global":
        X1 = torch.randn((model.n_samples, model.nb_cov_inflation + 1))
        X2 = torch.randn((model.n_samples, model.nb_cov_inflation - 1))
        with pytest.raises(RuntimeError):
            model.predict_prob_inflation(X1)
        with pytest.raises(RuntimeError):
            model.predict_prob_inflation(X1)


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_fail_predict(model):
    nb_plus = model.nb_cov + 1
    nb_moins = model.nb_cov - 1 if model.nb_cov < 0 else 0
    X1 = torch.randn((model.n_samples, nb_plus))
    X2 = torch.randn((model.n_samples, nb_moins))
    if model.nb_cov == 0:
        expected_error = AttributeError
    else:
        expected_error = RuntimeError
    with pytest.raises(expected_error):
        model.predict(X1)
    with pytest.raises(expected_error):
        model.predict(X2)


@pytest.mark.parametrize("zi", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["ZIPln"])
def test_transform(zi):
    z = zi.transform()
    assert z.shape == zi.endog.shape
    z, w = zi.transform(return_latent_prob=True)
    assert z.shape == w.shape == zi.endog.shape


def train_zi(formula):
    n_samples = 100
    if formula == "global":
        nb_cov_inflation = 0
        add_const_inflation = False
    elif formula in ["column-wise", "row-wise"]:
        nb_cov_inflation = 1
        add_const_inflation = True
    print("zero inflation", formula)
    zipln_param = get_simulation_parameters(
        zero_inflation_formula=formula,
        n_samples=n_samples,
        nb_cov_inflation=nb_cov_inflation,
        dim=80,
        add_const_inflation=add_const_inflation,
    )
    zipln_param._coef += 4
    endog = sample_zipln(zipln_param, seed=0)
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
    zi.fit(do_smart_init=False)
    return zi, zipln_param


def test_ziglobal():
    zi, zipln_param = train_zi("global")
    assert mae(zi.coef_inflation - zipln_param.coef_inflation) < 0.5
    assert mae(zi.proba_inflation - zipln_param.proba_inflation) < 0.1
    assert mae(zi.coef - zipln_param.coef) < 0.3
    assert mae(zi.covariance - zipln_param.covariance) < 0.3


def test_zicolumn():
    zi, zipln_param = train_zi("column-wise")
    assert mae(zi.coef_inflation - zipln_param.coef_inflation) < 6
    assert mae(zi.proba_inflation - zipln_param.proba_inflation) < 0.16
    assert mae(zi.coef - zipln_param.coef) < 0.5
    assert mae(zi.covariance - zipln_param.covariance) < 0.5


def test_zirow():
    zi, zipln_param = train_zi("row-wise")
    assert mae(zi.proba_inflation - zipln_param.proba_inflation) < 0.15
    assert mae(zi.coef - zipln_param.coef) < 0.8
    assert mae(zi.covariance - zipln_param.covariance) < 0.7
