import pytest
import pandas as pd
import torch

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models


@pytest.mark.parametrize("pln", dict_fixtures["all_pln"])
def test_data_setter_with_torch(pln):
    pln.counts = pln.counts
    pln.covariates = pln.covariates
    pln.offsets = pln.offsets
    pln.fit()


@pytest.mark.parametrize("pln", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_parameters_setter_with_torch(pln):
    pln.latent_mean = pln.latent_mean
    pln.latent_sqrt_var = pln.latent_sqrt_var
    pln.coef = pln.coef
    if pln._NAME == "PlnPCA":
        pln.components = pln.components
    pln.fit()


@pytest.mark.parametrize("pln", dict_fixtures["all_pln"])
def test_data_setter_with_numpy(pln):
    np_counts = pln.counts.numpy()
    if pln.covariates is not None:
        np_covariates = pln.covariates.numpy()
    else:
        np_covariates = None
    np_offsets = pln.offsets.numpy()
    pln.counts = np_counts
    pln.covariates = np_covariates
    pln.offsets = np_offsets
    pln.fit()


@pytest.mark.parametrize("pln", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_parameters_setter_with_numpy(pln):
    np_latent_mean = pln.latent_mean.numpy()
    np_latent_sqrt_var = pln.latent_sqrt_var.numpy()
    if pln.coef is not None:
        np_coef = pln.coef.numpy()
    else:
        np_coef = None
    pln.latent_mean = np_latent_mean
    pln.latent_sqrt_var = np_latent_sqrt_var
    pln.coef = np_coef
    if pln._NAME == "PlnPCA":
        pln.components = pln.components.numpy()
    pln.fit()


@pytest.mark.parametrize("pln", dict_fixtures["all_pln"])
def test_data_setter_with_pandas(pln):
    pd_counts = pd.DataFrame(pln.counts.numpy())
    if pln.covariates is not None:
        pd_covariates = pd.DataFrame(pln.covariates.numpy())
    else:
        pd_covariates = None
    pd_offsets = pd.DataFrame(pln.offsets.numpy())
    pln.counts = pd_counts
    pln.covariates = pd_covariates
    pln.offsets = pd_offsets
    pln.fit()


@pytest.mark.parametrize("pln", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_parameters_setter_with_pandas(pln):
    pd_latent_mean = pd.DataFrame(pln.latent_mean.numpy())
    pd_latent_sqrt_var = pd.DataFrame(pln.latent_sqrt_var.numpy())
    if pln.coef is not None:
        pd_coef = pd.DataFrame(pln.coef.numpy())
    else:
        pd_coef = None
    pln.latent_mean = pd_latent_mean
    pln.latent_sqrt_var = pd_latent_sqrt_var
    pln.coef = pd_coef
    if pln._NAME == "PlnPCA":
        pln.components = pd.DataFrame(pln.components.numpy())
    pln.fit()


@pytest.mark.parametrize("pln", dict_fixtures["all_pln"])
def test_fail_data_setter_with_torch(pln):
    with pytest.raises(ValueError):
        pln.counts = pln.counts - 100

    n, p = pln.counts.shape
    if pln.covariates is None:
        d = 0
    else:
        d = pln.covariates.shape[-1]
    with pytest.raises(ValueError):
        pln.counts = torch.zeros(n + 1, p)
    with pytest.raises(ValueError):
        pln.counts = torch.zeros(n, p + 1)

    with pytest.raises(ValueError):
        pln.covariates = torch.zeros(n + 1, d)

    with pytest.raises(ValueError):
        pln.offsets = torch.zeros(n + 1, p)

    with pytest.raises(ValueError):
        pln.offsets = torch.zeros(n, p + 1)


@pytest.mark.parametrize("pln", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_fail_parameters_setter_with_torch(pln):
    n, dim_latent = pln.latent_mean.shape
    dim = pln.counts.shape[1]

    with pytest.raises(ValueError):
        pln.latent_mean = torch.zeros(n + 1, dim_latent)

    with pytest.raises(ValueError):
        pln.latent_mean = torch.zeros(n, dim_latent + 1)

    with pytest.raises(ValueError):
        pln.latent_sqrt_var = torch.zeros(n + 1, dim_latent)

    with pytest.raises(ValueError):
        pln.latent_sqrt_var = torch.zeros(n, dim_latent + 1)

    if pln._NAME == "PlnPCA":
        with pytest.raises(ValueError):
            pln.components = torch.zeros(dim, dim_latent + 1)

        with pytest.raises(ValueError):
            pln.components = torch.zeros(dim + 1, dim_latent)

        if pln.covariates is None:
            d = 0
        else:
            d = pln.covariates.shape[-1]
        with pytest.raises(ValueError):
            pln.coef = torch.zeros(d + 1, dim)

        with pytest.raises(ValueError):
            pln.coef = torch.zeros(d, dim + 1)
