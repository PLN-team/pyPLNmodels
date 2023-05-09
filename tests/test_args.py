import os

from pyPLNmodels.models import PLN, PLNPCA, _PLNPCA
from pyPLNmodels import get_simulated_count_data, get_real_count_data
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import pandas as pd
import numpy as np

(
    counts_sim,
    covariates_sim,
    offsets_sim,
) = get_simulated_count_data(nb_cov=2)

couts_real = get_real_count_data(n_samples=298, dim=101)
RANKS = [2, 8]


@pytest.fixture
def instance_plnpca():
    plnpca = PLNPCA(ranks=RANKS)
    return plnpca


@pytest.fixture
def instance__plnpca():
    model = _PLNPCA(rank=RANKS[0])
    return model


@pytest.fixture
def instance_pln_full():
    return PLN()


all_instances = [lf("instance_plnpca"), lf("instance__plnpca"), lf("instance_pln_full")]


@pytest.mark.parametrize("instance", all_instances)
def test_pandas_init(instance):
    instance.fit(
        pd.DataFrame(counts_sim.numpy()),
        pd.DataFrame(covariates_sim.numpy()),
        pd.DataFrame(offsets_sim.numpy()),
    )


@pytest.mark.parametrize("instance", all_instances)
def test_numpy_init(instance):
    instance.fit(counts_sim.numpy(), covariates_sim.numpy(), offsets_sim.numpy())


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_only_counts(sim_pln):
    sim_pln.fit()


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_only_counts_and_offsets(sim_pln):
    sim_pln.fit(counts=counts_sim, offsets=offsets_sim)


@pytest.mark.parametrize("sim_pln", simulated_any_pln)
def test_only_Y_and_cov(sim_pln):
    sim_pln.fit(counts=counts_sim, covariates=covariates_sim)
