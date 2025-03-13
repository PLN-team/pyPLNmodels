# pylint: skip-file
import pytest

import torch

from pyPLNmodels import Pln, load_scrna, ZIPln, ZIPlnPCA, PlnMixture
from pyPLNmodels.calculations.elbos import (
    elbo_pln,
    elbo_zipln,
    elbo_ziplnpca,
    profiled_elbo_zipln,
    elbo_pln_diag,
    weighted_elbo_pln_diag,
    per_sample_elbo_pln_mixture_diag,
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
    zipca = ZIPlnPCA(data["endog"])
    zipca.fit()
    dirac = ~zipca._dirac
    with pytest.raises(RuntimeError):
        elbo_ziplnpca(
            endog=zipca._endog,
            marginal_mean=zipca._marginal_mean,
            offsets=zipca._offsets,
            latent_mean=zipca._latent_mean,
            latent_sqrt_variance=zipca._latent_sqrt_variance,
            latent_prob=zipca._latent_prob,
            components=zipca._components,
            marginal_mean_inflation=zipca._marginal_mean_inflation,
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


def test_weighted_elbo_pln_diag():
    data = load_scrna(n_samples=30)
    mixt = PlnMixture(data["endog"], n_cluster=2)
    mixt.fit()
    elbo_full = per_sample_elbo_pln_mixture_diag(
        endog=mixt._endog,
        marginal_means=mixt._marginal_means,
        offsets=mixt._offsets,
        latent_means=mixt._latent_means,
        latent_sqrt_variances=mixt._latent_sqrt_variances,
        diag_precisions=1 / (mixt._covariances),
    )
    elbo_full = torch.sum(elbo_full * (mixt._latent_prob.T))
    elbo_each = 0
    for k in range(mixt.n_cluster):
        elbo_k = weighted_elbo_pln_diag(
            endog=mixt._endog,
            marginal_mean=mixt._marginal_means[k],
            offsets=mixt._offsets,
            latent_mean=mixt._latent_means[k],
            latent_sqrt_variance=mixt._latent_sqrt_variances[k],
            diag_precision=1 / (mixt._covariances[k]),
            latent_prob=mixt._latent_prob[:, k],
        )
        elbo_each += elbo_k
    assert torch.allclose(elbo_each, elbo_full)
