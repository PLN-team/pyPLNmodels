import pytest

from pyPLNmodels.models import Pln, PlnPCA, PlnPCAcollection, ZIPln
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
