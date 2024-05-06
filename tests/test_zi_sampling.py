import torch
import pytest

from pyPLNmodels import (
    PlnParameters,
    ZIPlnParameters,
    get_simulation_parameters,
    sample_zipln,
    sample_zipln,
)


N_SAMPLES: int = 50
DIM: int = 25
NB_COV_INFLATION: int = 2
NB_COV: int = 1


def test_warning_get_simulation_parameters():
    with pytest.warns(UserWarning):
        param = get_simulation_parameters(add_const_inflation=True, dim=100)


def test_get_simulation_parameters_column_wise():
    param = get_simulation_parameters(
        zero_inflation_formula="column-wise",
        nb_cov_inflation=NB_COV_INFLATION,
        add_const_inflation=True,
    )
    assert param.exog_inflation.shape == (N_SAMPLES, NB_COV_INFLATION + 1)
    assert param.coef_inflation.shape == (NB_COV_INFLATION + 1, DIM)


def test_get_simulation_parameters_row_wise():
    param = get_simulation_parameters(
        zero_inflation_formula="row-wise",
        nb_cov_inflation=NB_COV_INFLATION,
        add_const_inflation=True,
    )
    assert param.coef_inflation.shape == (N_SAMPLES, NB_COV_INFLATION + 1)
    assert param.exog_inflation.shape == (NB_COV_INFLATION + 1, DIM)


def test_get_simulation_parameters_global():
    param = get_simulation_parameters(
        zero_inflation_formula="global", nb_cov_inflation=0, add_const_inflation=0
    )
    assert param.exog_inflation is None
    assert param.proba_inflation < 1 and param.proba_inflation > 0


def test_fails_get_simulation_parameters_global():
    with pytest.raises(ValueError):
        _ = get_simulation_parameters(
            zero_inflation_formula="global",
            nb_cov_inflation=NB_COV_INFLATION,
            add_const_inflation=True,
        )
        _ = get_simulation_parameters(
            zero_inflation_formula="global",
            nb_cov_inflation=0,
            add_const_inflation=True,
        )


def test_get_simulation_parameters_right_class():
    param = get_simulation_parameters(
        add_const_inflation=True, zero_inflation_formula="column-wise"
    )
    assert param.__class__.__name__ == "ZIPlnParameters"
