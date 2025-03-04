from typing import Dict, Any, Union, Iterable, Optional, List

import torch
import pandas as pd
import numpy as np

from pyPLNmodels.utils._data_handler import _handle_data, _extract_data_from_formula
from pyPLNmodels.utils._utils import _add_doc, _nice_string_of_dict
from pyPLNmodels.utils._viz import _show_information_criterion
from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.models.plnpca import PlnPCA


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnPCAcollection:
    """
    A collection of `PlnPCA` models, each with a different number of components.
    For more details, see Chiquet, J., Mariadassou, M., Robin, S.
    “Variational inference for probabilistic Poisson PCA.” Annals of applied stats.

    Examples
    --------
    >>> from pyPLNmodels import PlnPCAcollection, load_scrna
    >>> data = load_scrna()
    >>> plnpcas = PlnPCAcollection.from_formula("endog ~ 1", data = data, ranks = [5,8, 12])
    >>> plnpcas.fit()
    >>> print(plnpcas)
    >>> plnpcas.show()
    >>> print(plnpcas.best_model())
    >>> print(plnpcas[5])

    See also
    --------
    :class:`~pyPLNmodels.PlnPCA`
    :func:`pyPLNmodels.PlnPCAcollection.from_formula`
    :func:`pyPLNmodels.PlnPCAcollection.__init__`
    """

    @_add_doc(
        BaseModel,
        params="""
            ranks : Iterable[int], optional(keyword-only)
                The range of ranks, by default `(3, 5)`.
            """,
        example="""
            >>> from pyPLNmodels import PlnPCAcollection, load_scrna
            >>> data = load_scrna()
            >>> pcas = PlnPCAcollection(endog = data["endog"], ranks = [4,6,8])
            >>> pcas.fit()
            >>> print(pcas.best_model())
        """,
        returns="""
            PlnPCAcollection
        """,
        see_also="""
        :func:`pyPLNmodels.PlnPCAcollection.from_formula`
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
        """
        Constructor for `PlnPCAcollection`.

        Returns
        -------
        PlnPCAcollection

        See also
        --------
        :class:`~pyPLNmodels.PlnPCA`
        :meth:`~pyPLNmodels.PlnPCAcollection.from_formula`
        """
        self._dict_models = {}
        (
            self._endog,
            self._exog,
            self._offsets,
            self.column_names_endog,
            self.column_names_exog,
        ) = _handle_data(
            endog,
            exog,
            offsets,
            compute_offsets_method,
            add_const,
        )
        self._fitted = False
        self._init_models(ranks)

    def _init_models(
        self,
        ranks: Iterable[int],
    ):
        """
        Method for initializing the models.

        Parameters
        ----------
        ranks : Iterable[int]
            The range of ranks.
        add_const : bool
            Whether to add a column of ones in the exogenous variable.

        """
        if isinstance(ranks, (Iterable, np.ndarray)):
            for rank in ranks:
                if isinstance(rank, (int, np.integer)):
                    self._dict_models[rank] = PlnPCA(
                        endog=self._endog,
                        exog=self._exog,
                        offsets=self._offsets,
                        rank=rank,
                        add_const=False,
                    )
                else:
                    raise TypeError(
                        "Please instantiate `ranks` with either a list "
                        "of integers or an integer."
                    )
        elif isinstance(ranks, (int, np.integer)):
            self._dict_models[ranks] = PlnPCA(
                endog=self._endog,
                exog=self._exog,
                offsets=self._offsets,
                rank=ranks,
                add_const=False,
            )
        else:
            raise TypeError(
                "Please instantiate with either a list of integers or an integer."
            )

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
            ranks : Iterable[int], optional(keyword-only)
                The range of ranks, by default `(3, 5)`.
            """,
        example="""
            >>> from pyPLNmodels import PlnPCAcollection, load_scrna
            >>> data = load_scrna()
            >>> pcas = PlnPCAcollection.from_formula("endog ~ 1 + labels", data, ranks = [4,6,8])
            >>> pcas.fit()
            >>> print(pcas.best_model())
        """,
        returns="""
            PlnPCAcollection
        """,
        see_also="""
        :class:`~pyPLNmodels.PlnPCA`,
        :func:`pyPLNmodels.PlnPCAcollection.__init__`
        """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        ranks: Optional[Iterable[int]] = (3, 5),
    ):  # pylint: disable=missing-function-docstring
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog=endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            ranks=ranks,
            add_const=False,
        )

    @property
    def exog(self) -> torch.Tensor:
        """
        Property representing the `exog`.

        Returns
        -------
        torch.Tensor
            The `exog`.
        """
        return self[self.ranks[0]].exog

    @property
    def offsets(self) -> torch.Tensor:
        """
        Property representing the `offsets`.

        Returns
        -------
        torch.Tensor
            The `offsets`.
        """
        return self[self.ranks[0]].offsets

    @property
    def endog(self) -> torch.Tensor:
        """
        Property representing the `endog`.

        Returns
        -------
        torch.Tensor
            The `endog`.
        """
        return self[self.ranks[0]].endog

    @property
    def coef(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the coefficients.

        Returns
        -------
        Dict[int, torch.Tensor]
            The coefficients.
        """
        return {model.rank: model.coef for model in self.values()}

    @property
    def components(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the components.

        Returns
        -------
        Dict[int, torch.Tensor]
            The components.
        """
        return {model.rank: model.components for model in self.values()}

    @property
    def latent_mean(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the latent means.

        Returns
        -------
        Dict[int, torch.Tensor]
            The latent means.
        """
        return {model.rank: model.latent_mean for model in self.values()}

    def values(self):
        """
        Models in the collection as a list.

        Returns
        -------
        ValuesView
            The models in the collection.
        """
        return self._dict_models.values()

    def items(self):
        """
        Get the key-value pairs of the models in the collection.

        Returns
        -------
        ItemsView
            The key-value pairs of the models.
        """
        return self._dict_models.items()

    def __getitem__(self, rank: int) -> Any:
        """
        Model with the specified rank.

        Parameters
        ----------
        rank : int
            The rank of the model.

        Returns
        -------
        Any
            The model with the specified rank.
        """
        return self._dict_models[rank]

    def __len__(self) -> int:
        """
        Number of models in the collection.

        Returns
        -------
        int
            The number of models in the collection.
        """
        return len(self._dict_models)

    def __iter__(self):
        """
        Iterate over the models in the collection.

        Returns
        -------
        Iterator
            Iterator over the models.
        """
        return iter(self._dict_models)

    def __contains__(self, rank: int) -> bool:
        """
        Check if a model with the specified rank exists in the collection.

        Parameters
        ----------
        rank : int
            The rank to check.

        Returns
        -------
        bool
            True if a model with the specified rank exists, False otherwise.
        """
        return rank in self.keys()

    def keys(self):
        """
        Get the ranks of the models in the collection.

        Returns
        -------
        KeysView
            The ranks of the models.
        """
        return self._dict_models.keys()

    def get(self, key: Any, default: Any) -> Any:
        """
        Get the model with the specified key, or return a default value if the key does not exist.

        Parameters
        ----------
        key : Any
            The key to search for.
        default : Any
            The default value to return if the key does not exist.

        Returns
        -------
        Any
            The model with the specified key, or the default value if the key does not exist.
        """
        if key in self:
            return self[key]
        return default

    @property
    def ranks(self) -> List[int]:
        """
        Property representing the ranks (of the covariance matrix) of each model in the collection.

        Returns
        -------
        List[int]
            The ranks.
        """
        return [model.rank for model in self.values()]

    def _print_beginning_message(self) -> str:
        """
        Method for printing the beginning message.

        Returns
        -------
        str
            The beginning message.
        """
        return f"Adjusting {len(self.ranks)} Pln models for PCA analysis \n"

    @property
    def dim(self) -> int:
        """
        Property representing the dimension.

        Returns
        -------
        int
            The dimension.
        """
        return self[self.ranks[0]].dim

    @property
    def nb_cov(self) -> int:
        """
        Property representing the number of `exog`.

        Returns
        -------
        int
            The number of `exog`.
        """
        return self[self.ranks[0]].nb_cov

    def _init_next_model_with_current_model(
        self, next_model: PlnPCA, current_model: PlnPCA
    ):
        """
        Initialize the next `PlnPCA` model with the parameters of the current `PlnPCA` model.

        Parameters
        ----------
        next_model : PlnPCA
            The next model to initialize.
        current_model : PlnPCA
            The current model.
        """
        next_model.coef = current_model.coef
        new_components = torch.zeros(self.dim, next_model.rank).to(DEVICE)
        new_components[:, : current_model.rank] = (current_model.components).to(DEVICE)
        next_model.components = new_components

    def fit(
        self,
        maxiter: int = 400,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL,
        verbose: bool = False,
    ):
        """
        Fit each model in the `PlnPCAcollection`.

        Parameters
        ----------
        maxiter : int, optional
            The maximum number of iterations to be done, by default 400.
        lr : float, optional(keyword-only)
            The learning rate, by default 0.01.
        tol : float, optional(keyword-only)
            The tolerance, by default 1e-6.
        verbose : bool, optional(keyword-only)
            Whether to print verbose output, by default `False`.

        Returns
        -------
        PlnPCACollection
        """
        self._print_beginning_message()
        for i in range(len(self.values())):
            model = self[self.ranks[i]]
            model.fit(
                maxiter=maxiter,
                lr=lr,
                tol=tol,
                verbose=verbose,
            )
            if i < len(self.values()) - 1:
                next_model = self[self.ranks[i + 1]]
                self._init_next_model_with_current_model(next_model, model)
        self._print_ending_message()
        return self

    def _print_ending_message(self):
        delimiter = "=" * 70
        bic = self.BIC
        aic = self.AIC
        print(
            f"{delimiter}\n\nDONE!\n"
            f"Best model (lower BIC): rank {list(bic.keys())[np.argmin(list(bic.values()))]}\n"
            f"    Best model(lower AIC): rank {list(aic.keys())[np.argmin(list(aic.values()))]}\n\n"
            f"{delimiter}\n"
        )

    @property
    def BIC(self) -> Dict[int, int]:
        """
        Property representing the BIC scores of the models in the collection.

        Returns
        -------
        Dict[int, int]
            The BIC scores of the models.
        """
        return {model.rank: int(model.BIC) for model in self.values()}

    @property
    def AIC(self) -> Dict[int, int]:
        """
        Property representing the AIC scores of the models in the collection.

        Returns
        -------
        Dict[int, int]
            The AIC scores of the models.
        """
        return {model.rank: int(model.AIC) for model in self.values()}

    def best_model(self, criterion: str = "AIC") -> Any:
        """
        Get the best model according to the specified criterion.

        Parameters
        ----------
        criterion : str, optional
            The criterion to use ('AIC' or 'BIC'), by default 'AIC'.

        Returns
        -------
        Any
            The best model.
        """
        if criterion not in ("BIC", "AIC"):
            raise ValueError(f"Unknown criterion {criterion}")

        if criterion == "BIC":
            criterion = self.BIC
        if criterion == "AIC":
            criterion = self.AIC
        best_rank = self.ranks[np.argmin(list(criterion.values()))]
        return self[best_rank]

    @property
    def loglike(self) -> Dict[int, float]:
        """
        Property representing the log-likelihoods of the models in the collection.

        Returns
        -------
        Dict[int, float]
            The log-likelihoods of the models.
        """
        return {model.rank: model.loglike for model in self.values()}

    def show(self):
        """
        Show a plot with BIC scores, AIC scores, and negative log-likelihoods of the models.
        """
        _show_information_criterion(bic=self.BIC, aic=self.AIC, loglikes=self.loglike)

    @property
    def _useful_methods_strings(self) -> str:
        """
        Property representing the useful methods.

        Returns
        -------
        str
            The string representation of the useful methods.
        """
        return [".show()", ".best_model()", ".keys()", ".items()", ".values()"]

    @property
    def _useful_attributes_string(self) -> str:
        """
        Property representing the useful attributes.

        Returns
        -------
        str
            The string representation of the useful attributes.
        """
        return [".BIC", ".AIC", ".loglikes"]

    def __repr__(self) -> str:
        """
        Return a string representation of the `PlnPCAcollection` object.

        Returns
        -------
        str
            The string representation of the `PlnPCAcollection` object.
        """
        nb_models = len(self)
        delimiter = "\n" + "-" * 70 + "\n"

        # Header
        to_print = (
            f"{delimiter}"
            f"Collection of {nb_models} PlnPCAcollection models with {self.dim} variables."
            f"{delimiter}"
        )

        # Ranks considered
        to_print += f" - Ranks considered: {self.ranks}\n"

        # BIC metric
        dict_bic = {"rank": "criterion"} | self.BIC
        rank_bic = self.best_model(criterion="BIC")._rank
        to_print += (
            f" - BIC metric:\n{_nice_string_of_dict(dict_bic, best_rank=rank_bic)}\n"
        )
        to_print += f"   Best model (lower BIC): {rank_bic}\n\n"

        # AIC metric
        dict_aic = {"rank": "criterion"} | self.AIC
        rank_aic = self.best_model(criterion="AIC")._rank
        to_print += (
            f" - AIC metric:\n{_nice_string_of_dict(dict_aic, best_rank=rank_aic)}\n"
        )
        to_print += f"   Best model (lower AIC): {rank_aic}"

        # Footer
        to_print += (
            f"{delimiter}"
            f"* Useful attributes\n"
            f"    {self._useful_attributes_string}\n"
            f"* Useful methods\n"
            f"    {self._useful_methods_strings}"
            f"{delimiter}"
        )

        return to_print
