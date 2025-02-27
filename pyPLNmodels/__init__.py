import importlib.metadata

from .load_data import (
    load_oaks,
    load_scrna,
    load_microcosm,
    load_crossover,
    load_crossover_per_species,
)

from .pln import Pln
from .plnpca import PlnPCA
from .plnpcacollection import PlnPCAcollection
from .zipln import ZIPln
from .ziplnpca import ZIPlnPCA
from .network import PlnNetwork
from .plndiag import PlnDiag
from .plnmixture import PlnMixture
from .plnar import PlnAR
from .plnlda import PlnLDA

from .sampling import (
    PlnSampler,
    PlnPCASampler,
    ZIPlnSampler,
    ZIPlnPCASampler,
    PlnNetworkSampler,
    PlnDiagSampler,
    PlnMixtureSampler,
    PlnARSampler,
    PlnLDASampler,
)

from ._utils import get_confusion_matrix, get_label_mapping
from ._viz import plot_confusion_matrix


__version__ = importlib.metadata.version("pyplnmodels")

__all__ = [
    "Pln",
    "PlnPCA",
    "PlnPCAcollection",
    "PlnMixture",
    "PlnSampler",
    "PlnPCASampler",
    "ZIPlnSampler",
    "ZIPln",
    "PlnNetwork",
    "PlnNetworkSampler",
    "PlnDiag",
    "PlnDiagSampler",
    "ZIPlnPCASampler",
    "ZIPlnPCA",
    "PlnARSampler",
    "PlnAR",
    "PlnLDA",
    "PlnLDASampler",
    "get_confusion_matrix",
    "plot_confusion_matrix",
    "get_label_mapping",
    "load_microcosm",
    "load_oaks",
    "load_crossover_per_species",
    "load_crossover",
    "load_scrna",
]
