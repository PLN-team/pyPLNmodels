# pylint: skip-file
import pytest

from pyPLNmodels import Pln, load_scrna, ZIPln
from pyPLNmodels.elbos import elbo_pln, elbo_zipln


def test_right_pln_elbo():
    data = load_scrna()
    pln = Pln(data["endog"])
    pln.fit()
    profiled_elbo = pln.compute_elbo()
    elbo = elbo_pln(
        endog=pln._endog,
        marginal_mean=pln._marginal_mean,
        offsets=pln._offsets,
        latent_mean=pln._latent_mean,
        latent_sqrt_variance=pln._latent_sqrt_variance,
        covariance=pln._covariance,
    )
    assert elbo == profiled_elbo


def test_error_dirac_zi():
    data = load_scrna()
    zi = ZIPln(data["endog"])
    zi.fit()
    dirac = ~zi._dirac
    with pytest.raises(RuntimeError):
        elbo_zipln(
            endog=zi._endog,
            marginal_mean=zi._marginal_mean,
            offsets=zi._offsets,
            latent_mean=zi._latent_mean,
            latent_sqrt_variance=zi._latent_sqrt_variance,
            latent_prob=zi._latent_prob,
            covariance=zi._covariance,
            marginal_mean_inflation=zi._marginal_mean_inflation,
            dirac=dirac,
        )
