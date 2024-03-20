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
    for_formula=False,
    cov_list=["site", "time"],
    get_interaction=False,
    remove_useless=True,
    return_names=False,
    min_perc=0.025,
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
    get_interaction: bool, optional (keyword-only)
        If True, will give the interactions between each variables. Default to False
    remove_useless: bool, optional (keyword-only)
        If True, will remove all the interaction terms that does not appear.
    min_perc: float, optional (keyword-only)
        The minimum percentage of appearance of feature (ASV) to be selected.
        If the ASV is present in less than min_perc, it will be removed.
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

    best = (endog > 0).mean(axis=0) > min_perc
    endog = endog.loc[:, best]
    # covariates = covariates.loc[:,best]
    # print('best', best.mean())
    # z

    if get_affil is True:
        affil_stream = pkg_resources.resource_stream(
            __name__, "data/microcosm/affiliations.tsv"
        )
        affil = pd.read_csv(affil_stream, delimiter="\t")
    formula = "0 +"
    if get_interaction is True:
        separator = "* "
    else:
        separator = "+ "
    for i, key in enumerate(cov_list):
        formula += key
        if i < len(cov_list) - 1:
            formula += separator
    dm = dmatrix(formula, covariates)
    exog = pd.DataFrame(dm, columns=dm.design_info.column_names)
    if len(cov_list) > 0:
        exog = exog.iloc[:, 1:]
    if remove_useless:
        non_zero_cols = (exog.sum(axis=0) > 0).values
        exog = exog.loc[:, non_zero_cols]
    data = {"endog": endog.astype(float)}
    data["exog"] = exog
    if get_affil:
        data["affiliations"] = affil
    if for_formula:
        return data
    if get_affil:
        return data["endog"], data["exog"], data["affiliations"]
    if return_names is True:
        return data["endog"], data["exog"], covariates
    return data["endog"], data["exog"]
