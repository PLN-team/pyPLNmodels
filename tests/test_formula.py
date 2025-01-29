# pylint: skip-file

from patsy import PatsyError
import pytest

from pyPLNmodels import PlnPCA, load_microcosm


def test_wrong_formula():
    data = load_microcosm()
    with pytest.raises(PatsyError):
        pca = PlnPCA.from_formula("endog ~ 1 + exog**exog", data=data)
