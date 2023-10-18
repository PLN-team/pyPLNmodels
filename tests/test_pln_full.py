import pytest

from tests.conftest import dict_fixtures
from tests.utils import filter_models


@pytest.mark.parametrize("fitted_pln", dict_fixtures["fitted_model"])
@filter_models(["Pln"])
def test_number_of_iterations_pln_full(fitted_pln):
    nb_iterations = len(fitted_pln._elbos_list)
    assert 20 < nb_iterations < 2000


@pytest.mark.parametrize("pln", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["Pln"])
def test_latent_variables(pln):
    assert pln.transform().shape == pln.endog.shape
