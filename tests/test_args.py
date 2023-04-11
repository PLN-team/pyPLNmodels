from pyPLNmodels.VEM import PLN, PLNPCA
from pyPLNmodels import get_simulated_count_data
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import pandas as pd
import numpy as np

(
    counts_sim,
    covariates_sim,
    offsets_sim,
) = get_simulated_count_data()


RANKS = [4, 8]


@pytest.fixture
def my_instance_plnpca():
    plnpca = PLNPCA(ranks=RANKS)
    return plnpca


def test_pandas_init(my_instance_plnpca):
    my_instance_plnpca.fit(
        pd.DataFrame(counts_sim.numpy()),
        pd.DataFrame(covariates_sim.numpy()),
        pd.DataFrame(offsets_sim.numpy()),
    )


def test_best_model(best_models):
    print(best_models)
