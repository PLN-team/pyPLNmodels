# pylint: skip-file
import pytest

from tests._init_functions import (
    _Pln_init,
    _PlnPCA_init,
    _ZIPln_init,
    _PlnDiag_init,
    _PlnNetwork_init,
    _ZIPlnPCA_init,
    _PlnMixture_init,
    _PlnAR_init,
    _PlnLDA_init,
)

list_init = [
    _Pln_init,
    _PlnPCA_init,
    _ZIPln_init,
    _PlnDiag_init,
    _PlnNetwork_init,
    _ZIPlnPCA_init,
    _PlnMixture_init,
    _PlnAR_init,
    _PlnLDA_init,
]


def test_wrong_init_models():
    for init_func in list_init:
        with pytest.raises(ValueError):
            init_func("dumb_init_method")
