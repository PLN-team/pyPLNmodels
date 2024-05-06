import torch
import pandas as pd
import pkg_resources
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, LabelEncoder
import numpy as np
from patsy import dmatrix

from pyPLNmodels._utils import threshold_samples_and_dim

pd.set_option("display.max_columns", None)


def load_microcosm(
    n_samples: int = 300,
    dim=200,
    *,
    get_affil=False,
    cov_list=["site", "time"],
    min_perc=0.025,
    remove_useless=True,
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
    cov_list: list, optional (keyword-only).
        List with the wanted covariates. Should be included in ["site", "lineage", "time", "animal", "Sample", "echantillon","name","short_ID"].
        Default is ["site", "lineage", "time"], the only making sense as covariates.
        They are one hot encoded.
    min_perc: float, optional (keyword-only)
        The minimum percentage of appearance of feature (ASV) to be selected.
        If the ASV is present in less than min_perc, it will be removed.
    remove_useless: bool, optional (keyword-only)
        If True, will remove the useless samples, i.e. samples
        that have only zero counts.
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
    covariates = pd.read_csv(cov_stream, delimiter="\t")[cov_list].iloc[:n_samples, :]

    best_var = (endog > 0).mean(axis=0) >= min_perc
    endog = endog.loc[:, best_var]
    non_zero_samples = endog.mean(axis=1) > 0
    if remove_useless is True:
        endog = endog.loc[non_zero_samples, :]

    data = {"endog": endog}
    for cov_name in cov_list:
        covariate = covariates[cov_name]
        if remove_useless is True:
            data[cov_name] = covariate[non_zero_samples]
        else:
            data[cov_name] = covariate

    if get_affil is True:
        affil_stream = pkg_resources.resource_stream(
            __name__, "data/microcosm/affiliations.tsv"
        )
        affil = pd.read_csv(affil_stream, delimiter="\t").loc[1:dim, :]
        data["affiliations"] = affil.iloc[best_var.values, :]

    return data
