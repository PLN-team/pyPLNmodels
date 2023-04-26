import pkg_resources
import pandas as pd


def load_oaks():
    """Oaks amplicon data set

    This data set gives the abundance of 114 taxa (66 bacterial OTU,
    48 fungal OTUs) in 116 samples (leafs).

    A 114 taxa by 116 samples offset matrix is also given, based on the total number of reads
    found in each sample, which depend on the technology used for either
    bacteria (16S) or fungi (ITS1).

    For each sample, 3 additional covariates (tree, dist2ground, orientation) are known.

    The data is provided as dictionary with the following keys
        counts          a 114 x 116 np.array of integer (counts)
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
    counts_stream = pkg_resources.resource_stream(__name__, "data/oaks/counts.csv")
    offsets_stream = pkg_resources.resource_stream(__name__, "data/oaks/offsets.csv")
    covariates_stream = pkg_resources.resource_stream(
        __name__, "data/oaks/covariates.csv"
    )
    counts = pd.read_csv(counts_stream)
    offsets = pd.read_csv(offsets_stream)
    covariates = pd.read_csv(covariates_stream)
    oaks = {
        "counts": counts.to_numpy(),
        "offsets": offsets.to_numpy(),
        "tree": covariates.tree.to_numpy(),
        "dist2ground": covariates.distTOground.to_numpy(),
        "orientation": covariates.orientation.to_numpy(),
    }
    return oaks
