# pylint: skip-file
import pytest

from pyPLNmodels import ZIPln, load_scrna, Pln, load_microcosm

from .generate_models import get_model


def test_no_exog_inflation():
    data = load_scrna()
    with pytest.raises(ValueError):
        zi = ZIPln.from_formula("endog ~ 0", data)

    with pytest.raises(ValueError):
        zi = ZIPln(data["endog"], exog=None, add_const=False, add_const_inflation=False)

    with pytest.raises(ValueError):
        zi = ZIPln(
            data["endog"],
            exog=None,
            add_const=True,
            exog_inflation=None,
            add_const_inflation=False,
        )

    with pytest.raises(ValueError):
        zi = get_model(
            "ZIPln", "formula", {"nb_cov_inflation": 0, "add_const_inflation": False}
        )

    with pytest.raises(ValueError):
        zi = get_model(
            "ZIPln", "explicit", {"nb_cov_inflation": 0, "add_const_inflation": False}
        )


def test_endog_neg():
    data = load_scrna()
    data["endog"][3, 4] = -1
    with pytest.raises(ValueError):
        pln = Pln(data["endog"])


def test_wrong_shape():
    data = load_scrna()
    data["endog"] = data["endog"].iloc[: data["endog"].shape[0] // 2, :]
    with pytest.raises(ValueError):
        pln = Pln(data["endog"], exog=data["labels_1hot"])


def test_too_much_cov():
    data = load_microcosm()
    data["endog"] = data["endog"].iloc[:3, :]
    data["site_1hot"] = data["site_1hot"].iloc[:3, :]
    # with pytest.raises(ValueError):
    pln = Pln(data["endog"], exog=data["site_1hot"])


def test_not_tensor():
    with pytest.raises(AttributeError):
        pln = Pln([0, 1, 2])


def test_wront_method_offsets():
    data = load_scrna()
    with pytest.raises(ValueError):
        pln = Pln(data["endog"], compute_offsets_method="nothing")
