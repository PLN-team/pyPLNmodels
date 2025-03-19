# pylint: skip-file
import numpy as np
import torch
import pytest
from pyPLNmodels.utils._utils import get_label_mapping
from pyPLNmodels.models import PlnMixture, PlnLDA
from pyPLNmodels.sampling import PlnMixtureSampler
from pyPLNmodels.utils._viz import plot_confusion_matrix
from pyPLNmodels.load_data import load_scrna


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

from tests.utils import _get_formula
from tests.generate_models import get_model


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


def test_get_formula_add_const_only():
    assert _get_formula(0, add_const=True) == "endog ~ 1"


def test_get_model():
    get_model(
        "Pln",
        "explicit",
        {"nb_cov": 0, "add_const": False},
    )
    get_model(
        "ZIPln",
        "explicit",
        {"nb_cov_inflation": 0, "add_const": False},
    )


def test_label_mapping():
    sampler = PlnMixtureSampler(nb_cov=0, dim=30, n_samples=200)
    endog = sampler.sample()
    mixt = PlnMixture(endog=endog, n_cluster=sampler.n_cluster)
    mixt.fit()
    pred = mixt.clusters
    label_mapping = get_label_mapping(torch.tensor(pred), sampler.clusters)
    for k in range(mixt.n_cluster):
        moyenne = torch.mean(
            torch.abs(
                (pred == k).float()
                - torch.tensor(sampler.clusters == label_mapping[k]).float()
            )
        )
        assert moyenne < 0.1

    label_mapping = get_label_mapping(pred, torch.tensor(sampler.clusters))
    for k in range(mixt.n_cluster):
        moyenne = torch.mean(
            torch.abs(
                (pred == k).float()
                - torch.tensor(sampler.clusters == label_mapping[k]).float()
            )
        )
        assert moyenne < 0.1

    lda = PlnLDA(
        endog, clusters=torch.nn.functional.one_hot(torch.tensor(sampler.clusters))
    ).fit()
    pred_lda = lda.predict_clusters(endog)
    plot_confusion_matrix(pred_lda, lda._exog_clusters.cpu())

    rna = load_scrna()
    lda = PlnLDA(rna["endog"], clusters=rna["labels_1hot"])
    lda.fit()
    plot_confusion_matrix(lda.predict_clusters(rna["endog"]), rna["labels"])
