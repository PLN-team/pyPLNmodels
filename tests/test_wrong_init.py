import pytest

from pyPLNmodels.models import Pln, PlnPCA, PlnPCAcollection, ZIPln
from pyPLNmodels import get_pln_simulated_count_data
import numpy as np

from tests.import_data import (
    data_real,
)


endog_real = data_real["endog"]


@pytest.mark.parametrize("pln_model", [Pln, PlnPCA, PlnPCAcollection, ZIPln])
def test_init_with_zeros_pln(pln_model):
    endog_with_zeros = np.copy(endog_real)
    endog_with_zeros[4, :] *= 0
    with pytest.raises(ValueError):
        model = pln_model(endog_with_zeros)


@pytest.mark.parametrize("pln_model", [Pln, PlnPCA, PlnPCAcollection, ZIPln])
def test_wrong_formula_with_no_exog(pln_model):
    endog, exog, offsets = get_pln_simulated_count_data(add_const=False, nb_cov=0)
    data = {"endog": endog, "exog": exog, "offsets": offsets}
    with pytest.raises(ValueError):
        model = pln_model.from_formula("endog ~ 1 + exog", data=data)
