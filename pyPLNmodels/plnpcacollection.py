from typing import Dict, Any

import torch


class PlnPCAcollection:
    """
    A collection of PlnPCA models, each with a different number of components.
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
    """

    @property
    def exog(self) -> torch.Tensor:
        """
        Property representing the exog.

        Returns
        -------
        torch.Tensor
            The exog.
        """
        return self[self.ranks[0]].exog

    @property
    def endog(self) -> torch.Tensor:
        """
        Property representing the endog.

        Returns
        -------
        torch.Tensor
            The endog.
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
        return rank in self._dict_models.keys()

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
