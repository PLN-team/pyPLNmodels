# pylint: skip-file
import pytest
import torch

from pyPLNmodels import ZIPln, load_scrna, Pln, load_microcosm, PlnPCA, ZIPlnPCA


from .generate_models import get_model, get_dict_models_unfit
from tests._init_functions import _Pln_init, _PlnPCA_init, _ZIPln_init


def test_no_exog_inflation():
    rna = load_scrna()
    for str_model, zi_model in {"ZIPln": ZIPln, "ZIPlnPCA": ZIPlnPCA}.items():
        with pytest.raises(ValueError):
            zi = zi_model.from_formula("endog ~ 0", rna)

        with pytest.raises(ValueError):
            zi = zi_model(
                rna["endog"], exog=None, add_const=False, add_const_inflation=False
            )

        with pytest.raises(ValueError):
            zi = zi_model(
                rna["endog"],
                exog=None,
                add_const=True,
                exog_inflation=None,
                add_const_inflation=False,
            )

        with pytest.raises(ValueError):
            zi = get_model(
                str_model,
                "formula",
                {"nb_cov_inflation": 0, "add_const_inflation": False},
            )

        with pytest.raises(ValueError):
            zi = get_model(
                str_model,
                "explicit",
                {"nb_cov_inflation": 0, "add_const_inflation": False},
            )
        with pytest.raises(ValueError):
            zi = get_model(str_model, "wrong formula", {})


def test_endog_neg():
    rna = load_scrna()
    rna["endog"][3, 4] = -1
    with pytest.raises(ValueError):
        pln = Pln(rna["endog"])


def test_wrong_shape():
    rna = load_scrna()
    rna["endog"] = rna["endog"].iloc[: rna["endog"].shape[0] // 2, :]
    with pytest.raises(ValueError):
        pln = Pln(rna["endog"], exog=rna["labels_1hot"])


def test_too_much_cov():
    micro = load_microcosm()
    micro["endog"] = micro["endog"].iloc[:3, :]
    micro["site_1hot"] = micro["site_1hot"].iloc[:3, :]
    with pytest.raises(ValueError):
        pln = Pln(micro["endog"], exog=micro["site_1hot"])


def test_not_tensor():
    with pytest.raises(AttributeError):
        pln = Pln([0, 1, 2])


def test_wront_method_offsets():
    rna = load_scrna()
    with pytest.raises(ValueError):
        pln = Pln(rna["endog"], compute_offsets_method="nothing")


def test_setter():
    rna = load_scrna()
    pca = PlnPCA(rna["endog"])
    pca.fit()
    pca.components = pca.components.numpy()
    with pytest.raises(ValueError):
        pca.components = pca.components[:, :3]
    with pytest.raises(ValueError):
        pca.coef = pca.coef[:, :4]
    pca.coef = pca.coef.numpy()


def test_wrong_init_models():
    with pytest.raises(ValueError):
        _Pln_init("wrong formula")
    with pytest.raises(ValueError):
        _PlnPCA_init("wrong formula")
    with pytest.raises(ValueError):
        _ZIPln_init("wrong formula")


def test_bad_elbo():
    unfit_models = get_dict_models_unfit()
    for model_name in unfit_models:
        for init_method in ["formula", "explicit"]:
            for model in unfit_models[model_name][init_method]:
                model.fit(maxiter=1)
                with torch.no_grad():
                    if model_name == "PlnMixture":
                        model._sqrt_covariances *= 0
                    else:
                        model._latent_sqrt_variance *= 0
                if model_name != "PlnAR":
                    with pytest.raises(ValueError):
                        model.fit()


def test_bad_fit():
    unfit_models = get_dict_models_unfit()
    for model_name in unfit_models:
        for init_method in ["formula", "explicit"]:
            for model in unfit_models[model_name][init_method]:
                with pytest.raises(RuntimeError):
                    print(model)
                with pytest.raises(ValueError):
                    model.fit(maxiter=0.4)
