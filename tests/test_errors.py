# pylint: skip-file
import pytest
import torch

from pyPLNmodels import ZIPln, load_scrna, Pln, load_microcosm, PlnPCA

from .generate_models import get_model
from .conftest import dict_fitted_models, dict_unfit_models


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


def test_bad_elbo():
    for model_name in dict_fitted_models:
        for init_method in ["formula", "explicit"]:
            for model in dict_fitted_models[model_name][init_method]:
                with torch.no_grad():
                    model._latent_sqrt_variance *= 0
                with pytest.raises(ValueError):
                    model.fit()


def test_bad_fit():
    for model_name in dict_unfit_models:
        for init_method in ["formula", "explicit"]:
            for model in dict_unfit_models[model_name][init_method]:
                with pytest.raises(RuntimeError):
                    print(model)
                with pytest.raises(ValueError):
                    model.fit(maxiter=0.4)


def test_too_much_components_viz():
    data = load_scrna()
    pca = PlnPCA(data["endog"])
    pca.fit()
    with pytest.raises(ValueError):
        pca.pca_pairplot(n_components=pca.dim + 10)


def test_not_fitted_viz():
    data = load_scrna()
    pca = PlnPCA(data["endog"])
    with pytest.raises(RuntimeError):
        pca.plot_expected_vs_true()


def test_setter():
    data = load_scrna()
    pca = PlnPCA(data["endog"])
    pca.fit()
    pca.components = pca.components.numpy()
    with pytest.raises(ValueError):
        pca.components = pca.components[:, :3]
    with pytest.raises(ValueError):
        pca.coef = pca.coef[:, :4]
    pca.coef = pca.coef.numpy()
