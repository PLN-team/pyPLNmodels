import pytest
import pandas as pd
import torch

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models

single_models = ["Pln", "PlnPCA", "ZIPln"]


@pytest.mark.parametrize("model", dict_fixtures["loaded_model"])
def test_data_setter_with_torch(model):
    model.endog = model.endog
    model.exog = model.exog
    model.offsets = model.offsets
    if model._NAME == "ZIPln":
        if model._zero_inflation_formula != "global":
            model.exog_inflation = model.exog_inflation
        else:
            model.exog_inflation = None
    model.fit()


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(single_models)
def test_parameters_setter_with_torch(model):
    model.latent_mean = model.latent_mean
    model.latent_sqrt_var = model.latent_sqrt_var
    if model._NAME != "Pln":
        model.coef = model.coef
    if model._NAME == "PlnPCA" or model._NAME == "ZIPln":
        model.components = model.components
    if model._NAME == "ZIPln":
        model.coef_inflation = model.coef_inflation
    model.fit()
    pd_latent_mean = pd.DataFrame(model.latent_mean.numpy())
    pd_latent_sqrt_var = pd.DataFrame(model.latent_sqrt_var.numpy())
    if model.coef is not None:
        pd_coef = pd.DataFrame(model.coef.numpy())
    else:
        pd_coef = None
    model.latent_mean = pd_latent_mean
    model.latent_sqrt_var = pd_latent_sqrt_var
    if model._NAME != "Pln":
        model.coef = pd_coef
    if model._NAME == "PlnPCA":
        model.components = pd.DataFrame(model.components.numpy())
    if model._NAME == "ZIPln":
        model.coef_inflation = pd.DataFrame(model.coef_inflation.numpy())
    model.fit()
    np_latent_mean = model.latent_mean.numpy()
    np_latent_sqrt_var = model.latent_sqrt_var.numpy()
    if model.coef is not None:
        np_coef = model.coef.numpy()
    else:
        np_coef = None
    model.latent_mean = np_latent_mean
    model.latent_sqrt_var = np_latent_sqrt_var
    if model._NAME != "Pln":
        model.coef = np_coef
    if model._NAME == "PlnPCA" or model._NAME == "ZIPln":
        model.components = model.components.numpy()
    if model._NAME == "ZIPln":
        model.coef_inflation = model.coef_inflation.numpy()
    model.fit()
    n, dim_latent = model.latent_mean.shape
    dim = model.endog.shape[1]

    with pytest.raises(ValueError):
        model.latent_mean = torch.zeros(n + 1, dim_latent)

    with pytest.raises(ValueError):
        model.latent_mean = torch.zeros(n, dim_latent + 1)

    with pytest.raises(ValueError):
        model.latent_sqrt_var = torch.zeros(n + 1, dim_latent)

    with pytest.raises(ValueError):
        model.latent_sqrt_var = torch.zeros(n, dim_latent + 1)

    if model._NAME == "PlnPCA":
        with pytest.raises(ValueError):
            model.components = torch.zeros(dim, dim_latent + 1)

        with pytest.raises(ValueError):
            model.components = torch.zeros(dim + 1, dim_latent)

        if model.exog is None:
            d = 0
        else:
            d = model.exog.shape[-1]
        if model._NAME != "Pln":
            with pytest.raises(ValueError):
                model.coef = torch.zeros(d + 1, dim)

            with pytest.raises(ValueError):
                model.coef = torch.zeros(d, dim + 1)
        if model._NAME == "ZIPln":
            if model._zero_inflation_formula != "global":
                with pytest.raises(ValueError):
                    model.coef_inflation = torch.zeros(
                        model.coef_inflation.shape[0] + 1, model.coef_inflation.shape[1]
                    )
                with pytest.raises(ValueError):
                    model.coef_inflation = torch.zeros(
                        model.coef_inflation.shape[0], model.coef_inflation.shape[1] + 1
                    )
            else:
                model.coef_inflation = torch.zeros(2, 2)


@pytest.mark.parametrize("model", dict_fixtures["all_model"])
def test_data_setter_with_numpy(model):
    np_endog = model.endog.numpy()
    if model.exog is not None:
        np_exog = model.exog.numpy()
    else:
        np_exog = None
    np_offsets = model.offsets.numpy()
    model.endog = np_endog
    model.exog = np_exog
    model.offsets = np_offsets
    if model._NAME == "ZIPln":
        if model._zero_inflation_formula != "global":
            model.exog_inflation = model.exog_inflation.numpy()
    model.fit()
    with pytest.raises(ValueError):
        model.batch_size = 10
    pd_endog = pd.DataFrame(model.endog.numpy())
    if model.exog is not None:
        pd_exog = pd.DataFrame(model.exog.numpy())
    else:
        pd_exog = None
    pd_offsets = pd.DataFrame(model.offsets.numpy())
    model.endog = pd_endog
    model.exog = pd_exog
    model.offsets = pd_offsets
    model.fit()
    with pytest.raises(ValueError):
        model.endog = -model.endog

    n, p = model.endog.shape
    if model.exog is None:
        d = 0
    else:
        d = model.exog.shape[-1]
    with pytest.raises(ValueError):
        model.endog = torch.zeros(n + 1, p)
    with pytest.raises(ValueError):
        model.endog = torch.zeros(n, p + 1)

    with pytest.raises(ValueError):
        model.exog = torch.zeros(n + 1, d)

    with pytest.raises(ValueError):
        model.offsets = torch.zeros(n + 1, p)

    with pytest.raises(ValueError):
        model.offsets = torch.zeros(n, p + 1)
    if model._NAME == "ZIPln":
        with pytest.raises(ValueError):
            if model._zero_inflation_formula == "global":
                model.exog_inflation = torch.Tensor([1.0])
            if model._zero_inflation_formula == "row-wise":
                exog_inflation = torch.randn(model.nb_cov_inflation, model.dim + 1)
                model.exog_inflation = exog_inflation
            if model._zero_inflation_formula == "column-wise":
                exog_inflation = torch.randn(
                    model.n_samples - 1, model.nb_cov_inflation
                )
                model.exog_inflation = exog_inflation
