from typing import Union, Iterable, Optional, Dict

import torch
import numpy as np
import pandas as pd

from pyPLNmodels.models.collection import Collection
from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.models.plnpca import PlnPCA
from pyPLNmodels.utils._utils import _add_doc, _init_next_model_pca
from pyPLNmodels.utils._data_handler import _extract_data_from_formula
from pyPLNmodels.utils._viz import _show_collection_and_explained_variance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnPCACollection(Collection):
    """
    A collection of `PlnPCA` models, each with a different number of components.
    The number of components can also be referred to as the number of PCs, or the
    rank of the covariance matrix.
    For more details, see Chiquet, J., Mariadassou, M., Robin, S.
    “Variational inference for probabilistic Poisson PCA.” Annals of applied stats.

    Examples
    --------
    >>> from pyPLNmodels import PlnPCACollection, load_scrna
    >>> data = load_scrna()
    >>> plnpcas = PlnPCACollection.from_formula("endog ~ 1", data = data, ranks = [5,8, 12])
    >>> plnpcas.fit()
    >>> print(plnpcas)
    >>> plnpcas.show()
    >>> print(plnpcas.best_model())
    >>> print(plnpcas[5])

    See also
    --------
    :class:`~pyPLNmodels.PlnPCA`
    :func:`pyPLNmodels.PlnPCACollection.from_formula`
    :func:`pyPLNmodels.PlnPCACollection.__init__`
    :class:`~pyPLNmodels.PlnNetworkCollection`
    :class:`~pyPLNmodels.ZIPlnPCACollection`
    :class:`~pyPLNmodels.PlnMixtureCollection`
    """

    _type_grid = int
    _grid_value_name = "rank"
    PlnModel = PlnPCA

    @_add_doc(
        Collection,
        params="""
            ranks : Iterable[int], optional(keyword-only)
                The range of ranks, by default `(3, 5)`.
            """,
        example="""
            >>> from pyPLNmodels import PlnPCACollection, load_scrna
            >>> data = load_scrna()
            >>> pcas = PlnPCACollection(endog = data["endog"], ranks = [4,6,8])
            >>> pcas.fit()
            >>> print(pcas.best_model())
        """,
        returns="""
            PlnPCACollection
        """,
        see_also="""
        :func:`pyPLNmodels.PlnPCACollection.from_formula`
        """,
    )
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
        ranks: Optional[Iterable[int]] = (3, 5),
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            endog=endog,
            grid=ranks,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
              ranks : Iterable[int], optional(keyword-only)
                The ranks (or number of PCs) that needs to be tested.
                By default (3, 5)
              """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        ranks: Optional[Iterable[int]] = (3, 5),
    ):  # pylint: disable=missing-function-docstring, arguments-differ
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog=endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            ranks=ranks,
            add_const=False,
        )

    def _instantiate_model(self, grid_value):
        return PlnPCA(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            rank=grid_value,
            add_const=False,
        )

    def _init_next_model_with_current_model(
        self, next_model: PlnPCA, current_model: PlnPCA
    ):
        next_model = _init_next_model_pca(next_model, current_model)

    @property
    def components(self) -> Dict[float, torch.Tensor]:
        """
        Property representing the components of each model in the collection.

        Returns
        -------
        Dict[int, torch.Tensor]
            The components.
        """
        return {key: value.components for key, value in self.items()}

    @property
    def ranks(self):
        """
        Property representing the ranks (of the covariance matrix) of each model in the collection.

        Returns
        -------
        List[int]
            The ranks.
        """
        return self.grid

    @_add_doc(
        Collection,
        example="""
            >>> from pyPLNmodels import PlnPCACollection, load_scrna
            >>> data = load_scrna()
            >>> pcas = PlnPCACollection(endog = data["endog"], ranks = [4,6,8])
            >>> pcas.fit()
        """,
        returns="""
        PlnPCACollection
        """,
    )
    def fit(
        self,
        maxiter: int = 400,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL,
        verbose: bool = False,
    ):
        return super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    @property
    def latent_mean(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the latent mean, for each model in the collection.

        Returns
        -------
        Dict[int, torch.Tensor]
            The latent means.
        """
        return {model.rank: model.latent_mean for model in self.values()}

    @property
    def latent_variance(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the latent variance, for each model in the collection.

        Returns
        -------
        Dict[int, torch.Tensor]
            The latent variances.
        """
        return {model.rank: model.latent_variance for model in self.values()}

    @_add_doc(
        Collection,
        example="""
            >>> from pyPLNmodels import PlnPCACollection, load_scrna
            >>> data = load_scrna()
            >>> pcas = PlnPCACollection(endog = data["endog"], ranks = [4,6,8])
            >>> pcas.fit()
            >>> print(pcas.best_model())
        """,
        returns="""
        PlnPCACollection
        """,
    )
    def best_model(self, criterion: str = "BIC") -> PlnPCA:
        return super().best_model(criterion=criterion)

    def show(self, figsize: tuple = (10, 10)):
        """
        Show a plot with BIC scores, AIC scores, and negative log-likelihoods of the models.
        Also show the explained variance pourcentage.

        Parameters
        ----------
        figsize : tuple of two positive floats.
            Size of the figure that will be created. By default (10,10)
        """
        absc_label = "Number of Principal Components (i.e. rank number)"
        _show_collection_and_explained_variance(
            self, figsize=figsize, absc_label=absc_label
        )
