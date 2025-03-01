# pylint: skip-file
import pytest

import torch

from pyPLNmodels import Pln, load_scrna, ZIPln
from pyPLNmodels.calculations.elbos import (
    elbo_pln,
    elbo_zipln,
    profiled_elbo_zipln,
    elbo_pln_diag,
)
from pyPLNmodels.calculations._closed_forms import (
    _closed_formula_covariance,
    _closed_formula_diag_covariance,
)


def test_right_pln_elbo():
    data = load_scrna()
    pln = Pln(data["endog"])
    pln.fit()
    diag_prec = torch.diag(torch.linalg.inv(pln._covariance))
    elbo_diag_cov = elbo_pln(
        endog=pln._endog,
        marginal_mean=pln._marginal_mean,
        offsets=pln._offsets,
        latent_mean=pln._latent_mean,
        latent_sqrt_variance=pln._latent_sqrt_variance,
        precision=torch.diag(diag_prec),
    )
    elbo_diag_fast = elbo_pln_diag(
        endog=pln._endog,
        marginal_mean=pln._marginal_mean,
        offsets=pln._offsets,
        latent_mean=pln._latent_mean,
        latent_sqrt_variance=pln._latent_sqrt_variance,
        diag_precision=diag_prec,
    )
    assert torch.allclose(elbo_diag_cov, elbo_diag_fast)
    profiled_elbo = pln.compute_elbo()
    elbo = elbo_pln(
        endog=pln._endog,
        marginal_mean=pln._marginal_mean,
        offsets=pln._offsets,
        latent_mean=pln._latent_mean,
        latent_sqrt_variance=pln._latent_sqrt_variance,
        precision=torch.linalg.inv(pln._covariance),
    )
    assert torch.allclose(elbo, profiled_elbo)


def test_right_zipln_elbo():
    data = load_scrna()
    zi = ZIPln(data["endog"])
    zi.fit()
    profiled_elbo = zi.compute_elbo()
    elbo = elbo_zipln(
        endog=zi._endog,
        marginal_mean=zi._marginal_mean,
        offsets=zi._offsets,
        latent_mean=zi._latent_mean,
        latent_sqrt_variance=zi._latent_sqrt_variance,
        latent_prob=zi._latent_prob,
        precision=torch.linalg.inv(zi._covariance),
        marginal_mean_inflation=zi._marginal_mean_inflation,
    )
    assert torch.allclose(elbo, profiled_elbo)


def test_error_dirac_zi():
    data = load_scrna()
    zi = ZIPln(data["endog"])
    zi.fit()
    dirac = ~zi._dirac
    with pytest.raises(RuntimeError):
        profiled_elbo_zipln(
            endog=zi._endog,
            exog=zi._exog,
            offsets=zi._offsets,
            latent_mean=zi._latent_mean,
            latent_sqrt_variance=zi._latent_sqrt_variance,
            latent_prob=zi._latent_prob,
            marginal_mean_inflation=zi._marginal_mean_inflation,
            dirac=dirac,
        )


def test_closed_formula():
    data = load_scrna()
    pln = Pln(data["endog"])
    pln.fit()
    closed = _closed_formula_covariance(
        pln.marginal_mean, pln.latent_mean, pln.latent_sqrt_variance, pln.n_samples
    )
    closed_diagonal = _closed_formula_diag_covariance(
        pln.marginal_mean, pln.latent_mean, pln.latent_sqrt_variance, pln.n_samples
    )
    assert torch.allclose(closed_diagonal, torch.diag(closed))
