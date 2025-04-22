from typing import Union, Iterable, Optional, Dict

import torch
import numpy as np
import pandas as pd

from pyPLNmodels.models.collection import Collection
from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.models.plnmixture import PlnMixture
from pyPLNmodels.utils._utils import _add_doc
from pyPLNmodels.utils._data_handler import _extract_data_from_formula
from pyPLNmodels.utils._viz import _show_collection_and_clustering_criterions


class PlnMixtureCollection(Collection):
    """
    A collection of `PlnMixture` models, each with a different number of clusters.
    For more details, see:
    J. Chiquet, M. Mariadassou, S. Robin: "The Poisson-Lognormal Model as a Versatile
    Framework for the Joint Analysis of Species Abundances."

    Examples
    --------
    >>> from pyPLNmodels import PlnMixtureCollection, load_scrna
    >>> data = load_scrna()
    >>> mixtures = PlnMixtureCollection.from_formula("endog ~ 0", data = data, n_clusters = [2,3,4])
    >>> mixtures.fit()
    >>> print(mixtures)
    >>> mixtures.show()
    >>> print(mixtures.best_model())
    >>> print(mixtures[3])

    See also
    --------
    :class:`~pyPLNmodels.PlnMixture`
    :func:`pyPLNmodels.PlnNetworkCollection.from_formula`
    :func:`pyPLNmodels.PlnNetworkCollection.__init__`
    :class:`~pyPLNmodels.PlnPCACollection`
    :class:`~pyPLNmodels.ZIPlnPCACollection`
    :class:`~pyPLNmodels.PlnNetworkCollection`
    """

    _type_grid = int
    _grid_value_name = "n_cluster"
    PlnModel = PlnMixture

    @_add_doc(
        Collection,
        params="""
            n_clusters : Iterable[int], optional(keyword-only)
                The range of clusters to test, by default `(2, 3, 4)`.
            """,
        example="""
            >>> from pyPLNmodels import PlnMixtureCollection, load_scrna
            >>> data = load_scrna()
            >>> mixtures = PlnMixtureCollection(endog = data["endog"], n_clusters = [2,3,4])
            >>> mixtures.fit()
            >>> print(mixtures.best_model())
        """,
        returns="""
            PlnMixtureCollection
        """,
        see_also="""
        :func:`pyPLNmodels.PlnMixtureCollection.from_formula`
        """,
    )
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = False,
        n_clusters: Optional[Iterable[int]] = (2, 3, 4),
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            endog=endog,
            grid=n_clusters,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
              n_clusters : Iterable[int], optional(keyword-only)
                The number of clusters (or components in Kmeans) that needs to be tested.
                By default (2, 3, 4)
              """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        n_clusters: Optional[Iterable[int]] = (2, 3, 4),
    ):  # pylint: disable=missing-function-docstring, arguments-differ
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog=endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            n_clusters=n_clusters,
            add_const=False,
        )

    def _instantiate_model(self, grid_value):
        return PlnMixture(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            n_cluster=grid_value,
            add_const=False,
        )

    def _init_next_model_with_current_model(
        self, next_model: PlnMixture, current_model: PlnMixture
    ):
        pass  # Can not give any insights on the next model

    @property
    def n_clusters(self):
        """
        Property representing the number of cluster of each model in the collection.

        Returns
        -------
        List[int]
            The number of clusters.
        """
        return self.grid

    @_add_doc(
        Collection,
        example="""
            >>> from pyPLNmodels import PlnMixtureCollection, load_scrna
            >>> data = load_scrna()
            >>> mixtures = PlnMixtureCollection(endog = data["endog"], n_clusters = [2,3,4])
            >>> mixtures.fit()
        """,
        returns="""
        PlnMixtureCollection
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
    def latent_means(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the latent means, for each model in the collection.

        Returns
        -------
        Dict[int, torch.Tensor]
            The latent means.
        """
        return {model.n_cluster: model.latent_means for model in self.values()}

    @property
    def latent_variances(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the latent mean, for each model in the collection.

        Returns
        -------
        Dict[int, torch.Tensor]
            The latent variances.
        """
        return {model.n_clusters: model.latent_variances for model in self.values()}

    def best_model(self, criterion: str = "BIC") -> PlnMixture:
        """
        Get the best model according to the specified criterion.

        Parameters
        ----------
        criterion : str, optional
            The criterion to use ('AIC', 'BIC' or 'silhouette'), by default 'silhouette'.

        Examples
        --------
        >>> from pyPLNmodels import PlnMixtureCollection, load_scrna
        >>> data = load_scrna()
        >>> mixtures = PlnMixtureCollection(endog = data["endog"], n_clusters = [2,3,4])
        >>> mixtures.fit()
        >>> print(mixtures.best_model())

        Returns
        -------
        PlnMixtureCollection
            The best model.
        """
        return super().best_model(criterion=criterion)

    @property
    def WCSS(self) -> Dict[int, int]:
        """
        Compute the Within-Cluster Sum of Squares on the latent positions
        for each model in the collection.

        The higher the better, but increasing n_cluster can only increase the
        metric. A trade-off (with the elbow method for example) must be applied.

        Returns
        -------
        Dict[int, float]
            The WCSS scores of the models.
        """
        return {grid_value: int(self[grid_value].WCSS) for grid_value in self.grid}

    @property
    def silhouette(self) -> Dict[int, int]:
        """
        Compute the silhouette score on the latent_positions for each model in the collection.
        See scikit-learn.metrics.silhouette_score for more information.

        The higher the better.

        Returns
        -------
        Dict[int, float]
            The silhouette scores of the models.
        """
        return {grid_value: self[grid_value].silhouette for grid_value in self.grid}

    def show(self, figsize: tuple = (15, 10)):
        """
        Show a plot with BIC scores, AIC scores, and negative log-likelihoods of the models.
        Also show two cluster criterion.

        Parameters
        ----------
        figsize : tuple of two positive floats.
            Size of the figure that will be created. By default (10,15)
        """
        absc_label = ""
        _show_collection_and_clustering_criterions(
            self, figsize=figsize, absc_label=absc_label
        )

    def _best_grid_value(self, criterion):
        possible_criterions = ("BIC", "AIC", "silhouette")
        if criterion not in possible_criterions:
            raise ValueError(
                f"Unknown criterion {criterion}. Availables: {possible_criterions}."
            )
        if criterion == "BIC":
            criterion = self.BIC
        elif criterion == "AIC":
            criterion = self.AIC
        elif criterion == "silhouette":
            criterion = self.silhouette
            criterion = {
                grid_value: -score for grid_value, score in self.silhouette.items()
            }
        return self.grid[np.argmin(list(criterion.values()))]
