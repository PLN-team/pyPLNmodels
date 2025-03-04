import pandas as pd
import pkg_resources

from .utils import _threshold_samples_and_dim


def load_microcosm(
    n_samples: int = 400,
    dim=100,
):
    """
    Get real count data from the microcosm
    (https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA875059) dataset.

    References:
        "Microbiota members from body sites of dairy cows are largely shared within
        individual hosts throughout lactation but sharing is limited in the herd" from
        Mahendra Mariadassou, Laurent X. Nouvel, Fabienne Constant, Diego P. Morgavi,
        Lucie Rault, Sarah Barbey, Emmanuelle Helloin, Olivier Rué, Sophie Schbath,
        Frederic Launay, Olivier Sandra, Rachel Lefebvre, Yves Le Loir, Pierre Germon,
        Christine Citti & Sergine Even
        https://animalmicrobiome.biomedcentral.com/articles/10.1186/s42523-023-00252-w

    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default 400.
    dim : int, optional
        Dimension, by default 100.

    Returns
    -------
    data: Dict
        The different (key, values) are:
            `endog` that represents the counts
            `site` and `site_1hot` that represent the site and the one-hot encoded site.
            `time` and `time_1hot` that represent the time and the one-hot encoded time.
            `lineage` and `lineage_1hot` that represent the lineage and the one-hot encoded lineage.

    Examples
    --------
    >>> from pyPLNmodels import load_microcosm
    >>> micro = load_microcosm()
    >>> print('Keys: ', micro.keys())
    >>> print(micro["endog"].head())
    >>> print(micro["endog"].describe())
    """
    max_samples = 921
    max_dim = 1209
    n_samples, dim = _threshold_samples_and_dim(max_samples, max_dim, n_samples, dim)
    endog_stream = pkg_resources.resource_stream(__name__, "data/microcosm/counts.tsv")
    endog = (
        (pd.read_csv(endog_stream, delimiter="\t"))
        .drop(columns="Sample")
        .iloc[:n_samples, :dim]
    )
    cov_stream = pkg_resources.resource_stream(__name__, "data/microcosm/metadata.tsv")
    covariates = pd.read_csv(cov_stream, delimiter="\t").iloc[:n_samples, :]
    affil_stream = pkg_resources.resource_stream(
        __name__, "data/microcosm/affiliations.tsv"
    )

    data = {"endog": endog}
    cov_list = ["site", "lineage", "time"]
    for cov_name in cov_list:
        covariate = covariates[cov_name].squeeze()
        data[cov_name + "_1hot"] = pd.get_dummies(covariate)
        data[cov_name] = covariate

    data["affiliations"] = pd.read_csv(affil_stream, delimiter="\t").loc[1:dim, :]

    return data
