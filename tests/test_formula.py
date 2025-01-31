# pylint: skip-file
from patsy import PatsyError
import pytest

from pyPLNmodels import PlnPCA, ZIPln, load_microcosm


def test_wrong_formula():
    micro = load_microcosm()
    with pytest.raises(PatsyError):
        PlnPCA.from_formula("endog ~ 1 + exog**exog", data=micro)


def test_wrong_formula_infla():
    micro = load_microcosm()
    with pytest.raises(PatsyError):
        ZIPln.from_formula("endog ~ 1 | exog**exog", data=micro)
