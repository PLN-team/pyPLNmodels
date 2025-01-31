import pandas as pd
import pkg_resources

from .utils import _threshold_samples_and_dim


def load_microcosm(
    n_samples: int = 300,
    dim=50,
):
    """
    Get real count data from the microcosm
    (https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA875059) dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default 300.
    dim : int, optional
        Dimension, by default 50.

    Returns
    -------
    Dict:
        Dictionary. The different key, values are:
            `endog` that reprensents the counts
            `site` and `site_1hot` that represent the site and the one hot encoded site.
            `time` and `time_1hot` that represent the time and the one hot encoded time.
            `lineage` and `lineage_1hot` that represent the lineage and the one hot lineage site.
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
