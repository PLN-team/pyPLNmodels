import pkg_resources
import pandas as pd
import numpy as np


def load_oaks():
    """Oaks amplicon data set

    This data set gives the abundance of 114 taxa (66 bacterial OTUs,
    48 fungal OTUs) in 116 samples (leaves).

    A 114 taxa by 116 samples offset matrix is also given, based on the total number of reads
    found in each sample, which depends on the technology used for either
    bacteria (16S) or fungi (ITS1).

    For each sample, 3 additional `exog` (tree, dist2ground, orientation) are known.

    The data is provided as a dictionary with the following keys:
        `endog`          a 114 x 116 np.array of integers (endog)
        `offsets`        a 114 x 116 np.array of integers (offsets)
        `tree`           a 114 x 1 vector of characters for the tree status
                         with respect to the pathogen (susceptible, intermediate, or resistant)
        `dist2ground`    a 114 x 1 vector encoding the distance of the sampled leaf
                         to the base of the ground
        `orientation`    a 114 x 1 vector encoding the orientation of the
                         branch (South-West SW or North-East NE)

    Source: data from B. Jakuschkin and coauthors.

    References:

     Jakuschkin, B., Fievet, V., Schwaller, L. et al. Deciphering the
     Pathobiome: Intra- and Interkingdom Interactions Involving the
     Pathogen Erysiphe alphitoides. Microb Ecol 72, 870–880 (2016).
     doi:10.1007/s00248-016-0777-x

    Returns
    -------
    oaks: Dict
        The different (key, values) are:
            `endog` a 114 x 116 np.array of integers (endog)
            `offsets` a 114 x 116 np.array of integers (offsets)
            `tree` a 114 x 1 vector of characters for the tree status
                   with respect to the pathogen (susceptible, intermediate, or resistant)
            `dist2ground` a 114 x 1 vector encoding the distance of the sampled leaf
                          to the base of the ground
            `orientation` a 114 x 1 vector encoding the orientation of the
                          branch (South-West SW or North-East NE)

    Examples
    --------
    >>> from pyPLNmodels import load_oaks
    >>> oaks = load_oaks()
    >>> print('Keys: ', oaks.keys())
    >>> print(oaks["endog"].head())
    >>> print(oaks["endog"].describe())
    """
    endog = pd.read_csv(pkg_resources.resource_stream(__name__, "data/oaks/counts.csv"))
    offsets = pd.read_csv(
        pkg_resources.resource_stream(__name__, "data/oaks/offsets.csv")
    )
    exog = pd.read_csv(
        pkg_resources.resource_stream(__name__, "data/oaks/covariates.csv")
    )
    oaks = {
        "endog": endog,
        "offsets": np.log(offsets),
        "tree": exog.tree.squeeze(),
        "dist2ground": exog.distTOground.squeeze(),
        "orientation": exog.orientation.squeeze(),
    }
    return oaks
