import torch


from pyPLNmodels import get_simulation_parameters, get_pln_simulated_count_data
from .test_zi_sampling import N_SAMPLES, DIM, NB_COV


def test_get_simulation_parameters_empty():
    param = get_simulation_parameters()
    assert param.exog.shape == (N_SAMPLES, NB_COV + 1)


def test_get_simulation_parameters_no_add_const():
    param = get_simulation_parameters(add_const=False)
    assert param.exog.shape == (N_SAMPLES, NB_COV)
    assert param.__class__.__name__ == "PlnParameters"


def test_get_pln_data():
    length_3_tupple = get_pln_simulated_count_data()
    assert len(length_3_tupple) == 3
    endog = length_3_tupple[0]
    exog = length_3_tupple[1]
    offsets = length_3_tupple[2]
    assert endog.shape == (N_SAMPLES, DIM)
    assert exog.shape == (N_SAMPLES, NB_COV + 1)
    assert offsets.shape == (N_SAMPLES, DIM)


def test_get_pln_data_with_true_param():
    length_5_tupple = get_pln_simulated_count_data(return_true_param=True)
    assert len(length_5_tupple) == 5
    endog = length_5_tupple[0]
    exog = length_5_tupple[1]
    offsets = length_5_tupple[2]
    covariance = length_5_tupple[3]
    coef = length_5_tupple[4]
    assert endog.shape == (N_SAMPLES, DIM)
    assert exog.shape == (N_SAMPLES, NB_COV + 1)
    assert offsets.shape == (N_SAMPLES, DIM)
    assert coef.shape == (NB_COV + 1, DIM)
    assert covariance.shape == (DIM, DIM)


def test_no_randomness():
    endog, _, _ = get_pln_simulated_count_data(seed=1)
    endog_bis, _, _ = get_pln_simulated_count_data(seed=1)
    assert torch.norm(endog_bis - endog_bis) < 1e-8


def test_no_exog():
    endog, exog, offsets = get_pln_simulated_count_data(nb_cov=0, add_const=False)
    assert exog == None


def test_get_simulation_parameters():
    param = get_simulation_parameters(nb_cov=0)
    assert param.exog.shape == (N_SAMPLES, 1)
    assert param.coef.shape == (1, DIM)
    param = get_simulation_parameters(nb_cov=3, add_const=False)
    assert param.exog.shape == (N_SAMPLES, 3)
    assert param.coef.shape == (3, DIM)
