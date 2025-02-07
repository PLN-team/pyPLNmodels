# pylint: skip-file
import pytest

from pyPLNmodels import PlnSampler, PlnPCASampler, ZIPlnSampler


def test_attributes():
    sampler = PlnSampler()
    assert sampler.n_samples == 100
    assert sampler.dim == 20
    assert sampler.exog.shape == (100, 2)
    assert sampler.covariance.shape == (20, 20)
    assert sampler.coef.shape == (2, 20)

    sampler.sample()

    sampler_0cov = PlnSampler(nb_cov=0, add_const=False)
    assert sampler_0cov.exog is None
    assert sampler_0cov.coef is None
    sampler_0cov.sample()


def test_offsets():
    sampler = PlnSampler(add_offsets=True)


def test_pca_sampler():
    sampler = PlnPCASampler()
    assert sampler.rank == 5
    assert sampler.covariance.shape == (20, 20)
    sampler.sample()


def test_sampler_zi():
    sampler = ZIPlnSampler()
    sampler.sample()
    with pytest.raises(ValueError):
        ZIPlnSampler(nb_cov_inflation=0, add_const_inflation=False)
    assert sampler.coef_inflation.shape == (2, 20)
