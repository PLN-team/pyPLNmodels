# pylint: skip-file
import pytest
import torch
import numpy as np

from pyPLNmodels.utils._data_handler import (
    _handle_data,
    _add_constant_to_exog,
    _check_full_rank_exog,
    _check_data_shapes,
)
from pyPLNmodels import load_oaks, load_microcosm, Pln, PlnSampler, PlnLDA, PlnMixture

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

oaks = load_oaks()


def test_handle_data_valid():
    endog, exog, offsets, col_names_endog, col_names_exog = _handle_data(
        oaks["endog"], oaks["dist2ground"], oaks["offsets"], "zero", True
    )
    assert isinstance(endog, torch.Tensor)
    assert isinstance(exog, torch.Tensor)
    assert isinstance(offsets, torch.Tensor)
    assert col_names_endog is not None
    assert col_names_exog is not None


def test_handle_data_invalid():
    with pytest.raises(ValueError):
        _handle_data(
            oaks["endog"],
            oaks["dist2ground"],
            torch.tensor([[0.1], [0.2]]),
            "zero",
            True,
        )


def test_handle_data_logsum():
    _, _, offsets, _, _ = _handle_data(
        oaks["endog"], oaks["dist2ground"], None, "logsum", True
    )
    assert isinstance(offsets, torch.Tensor)


def test_add_constant_to_exog_valid():
    exog_with_const = _add_constant_to_exog(
        torch.tensor(oaks["dist2ground"].values).unsqueeze(1).to(DEVICE),
        len(oaks["dist2ground"]),
    )
    assert exog_with_const.shape[1] == 2


def test_add_constant_to_exog_invalid():
    with pytest.raises(ValueError):
        _add_constant_to_exog(
            torch.tensor(oaks["dist2ground"].values), len(oaks["dist2ground"]) + 1
        )


def test_check_full_rank_exog_valid():
    _check_full_rank_exog(torch.tensor(oaks["dist2ground"].values).unsqueeze(1))


def test_check_full_rank_exog_invalid():
    with pytest.raises(ValueError):
        _check_full_rank_exog(torch.tensor([[1, 1], [1, 1]]).float())

    data = load_microcosm()
    with pytest.raises(ValueError):
        pln = PlnLDA.from_formula("endog~ 1| site ", data)

    with pytest.raises(ValueError):
        pln = PlnMixture.from_formula("endog~ 1", data, n_cluster=3)


def test_check_data_shapes_valid():
    _check_data_shapes(
        torch.tensor(oaks["endog"].values),
        torch.tensor(oaks["dist2ground"].values).unsqueeze(1),
        torch.tensor(oaks["offsets"].values),
    )


def test_check_data_shapes_invalid():
    with pytest.raises(ValueError):
        _check_data_shapes(
            torch.tensor(oaks["endog"].values),
            torch.tensor(oaks["dist2ground"].values),
            torch.tensor([[0.1], [0.2]]),
        )


def test_remove_column_names():
    micro = load_microcosm()
    micro["endog"].iloc[:, 6] *= 0
    pln = Pln(micro["endog"], exog=micro["site_1hot"], add_const=False)


def test_remove_column_names_exog():
    micro = load_microcosm()
    micro["site_1hot"].iloc[:, 1] *= 0
    micro["site_1hot"] = micro["site_1hot"].astype(np.float32)
    pln = Pln(micro["endog"], exog=micro["site_1hot"], add_const=False)
    sampler = PlnSampler()
    endog = sampler.sample()
    exog = sampler.exog
    exog[:, 0] *= 0
    pln = Pln(endog, exog=exog)
