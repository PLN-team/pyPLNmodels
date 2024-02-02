import torch


from pyPLNmodels import (
    PlnParameters,
    ZIPlnParameters,
    get_simulation_parameters,
    sample_zipln,
    sample_zipln,
)


N_SAMPLES: int = 100
DIM: int = 25
NB_COV: int = 1
NB_COV_INFLATION: int = 0


def test_get_simulation_parameters_pln():
    param = get_simulation_parameters()
    assert param.exog.shape == (N_SAMPLES, NB_COV + 1)
