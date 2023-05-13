import pytest
import pandas as pd

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models


@pytest.mark.parametrize("pln", dict_fixtures["all_pln"])
@filter_models(["PLN", "PLNPCA"])
def test_setter_with_numpy(pln):
    np_counts = pln.counts.numpy()
    pln.counts = np_counts
    pln.fit()


@pytest.mark.parametrize("pln", dict_fixtures["all_pln"])
@filter_models(["PLN", "PLNPCA"])
def test_setter_with_pandas(pln):
    pd_counts = pd.DataFrame(pln.counts.numpy())
    pln.counts = pd_counts
    pln.fit()
