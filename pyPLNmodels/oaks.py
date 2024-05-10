import pkg_resources
import pandas as pd
import numpy as np


def load_oaks():
    """Oaks amplicon data set

    This data set gives the abundance of 114 taxa (66 bacterial OTU,
    48 fungal OTUs) in 116 samples (leafs).

    A 114 taxa by 116 samples offset matrix is also given, based on the total number of reads
    found in each sample, which depend on the technology used for either
    bacteria (16S) or fungi (ITS1).

    For each sample, 3 additional exog (tree, dist2ground, orientation) are known.

    The data is provided as dictionary with the following keys
        endog          a 114 x 116 np.array of integer (endog)
        offsets         a 114 x 116 np.array of integer (offsets)
        tree            a 114 x 1 vector of character for the tree status with respect to the pathogen (susceptible, intermediate or resistant)
        dist2ground     a 114 x 1 vector encoding the distance of the sampled leaf to the base of the ground
        orientation     a 114 x 1 vector encoding the orientation of the branch (South-West SW or North-East NE)

    Source: data from B. Jakuschkin and coauthors.

    References:

     Jakuschkin, B., Fievet, V., Schwaller, L. et al. Deciphering the
     Pathobiome: Intra- and Interkingdom Interactions Involving the
     Pathogen Erysiphe alphitoides . Microb Ecol 72, 870–880 (2016).
     doi:10.1007/s00248-016-0777-x
    """
    endog_stream = pkg_resources.resource_stream(__name__, "data/oaks/counts.csv")
    offsets_stream = pkg_resources.resource_stream(__name__, "data/oaks/offsets.csv")
    exog_stream = pkg_resources.resource_stream(__name__, "data/oaks/covariates.csv")
    endog = pd.read_csv(endog_stream)
    offsets = pd.read_csv(offsets_stream)
    exog = pd.read_csv(exog_stream)
    oaks = {
        "endog": endog.to_numpy(),
        "offsets": np.log(offsets.to_numpy()),
        "tree": exog.tree.to_numpy(),
        "dist2ground": exog.distTOground.to_numpy(),
        "orientation": exog.orientation.to_numpy(),
    }
    return oaks
