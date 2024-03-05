import torch
import pandas as pd
import pkg_resources
from sklearn.preprocessing import OneHotEncoder

from pyPLNmodels._utils import threshold_samples_and_dim


def load_microcosm(
    n_samples: int = 300,
    dim=200,
    *,
    get_affil=False,
    for_formula=False,
    cov_list=["site", "lineage", "time"],
):
    """
    Get real count data from the microcosm
    (https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA875059) dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default max_samples.
    dim : int, optional
        Dimension, by default max_dim.
    get_affil: bool, optional (keyword-only)
        If True, will return the affiliations also. Default to False .
    for_formula: bool, optional(keyword-only)
        If True, will return a dict so that it can
        be passed into a formula.
    cov_list: list, optional (keyword-only).
        List with the wanted covariates. Should be included in ["site", "lineage", "time", "animal", "Sample", "echantillon","name","short_ID"].
        Default is ["site", "lineage", "time"], the only making sense as covariates.
        They are one hot encoded.
    """
    max_samples = 921
    max_dim = 1209
    n_samples, dim = threshold_samples_and_dim(max_samples, max_dim, n_samples, dim)
    endog_stream = pkg_resources.resource_stream(__name__, "data/microcosm/counts.tsv")
    endog = (
        (pd.read_csv(endog_stream, delimiter="\t"))
        .drop(columns="Sample")
        .iloc[:n_samples, :dim]
    )
    cov_stream = pkg_resources.resource_stream(__name__, "data/microcosm/metadata.tsv")
    cov = pd.read_csv(cov_stream, delimiter="\t")[cov_list].iloc[:n_samples, :]
    if get_affil is True:
        affil_stream = pkg_resources.resource_stream(
            __name__, "data/microcosm/affiliations.tsv"
        )
        affil = pd.read_csv(affil_stream, delimiter="\t")
    data = {}
    for name in cov.columns:
        encoder = OneHotEncoder(drop="first")
        hot = torch.from_numpy(encoder.fit_transform(cov).toarray())
        data[name] = hot
    data["endog"] = endog
    if get_affil:
        data["affiliations"] = affil
    if for_formula:
        return data
    endog = data["endog"]
    encoder = OneHotEncoder(drop="first")
    exog = encoder.fit_transform(cov).toarray()
    if get_affil:
        return endog, exog, affil
    return endog, exog
