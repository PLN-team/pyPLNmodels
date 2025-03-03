from typing import Optional, Union
import torch
import numpy as np
import pandas as pd

from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.calculations.elbos import elbo_plnpca
from pyPLNmodels.calculations._initialization import (
    _init_coef,
    _init_components,
    _init_latent_mean_pca,
    _init_latent_sqrt_variance_pca,
)
from pyPLNmodels.utils._data_handler import _extract_data_from_formula, _array2tensor
from pyPLNmodels.utils._utils import _add_doc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnPCA(BaseModel):
    """Principal Component Analysis on top of a PLN model, that is a PLN
    model with low-rank covariance, adapted to datasets with lots of features.
    For more details, see Chiquet, J., Mariadassou, M., Robin, S.
    “Variational inference for probabilistic Poisson PCA.” Annals of applied stats.

    Examples
    --------
    >>> from pyPLNmodels import PlnPCA, load_scrna
    >>> data = load_scrna()
    >>> pca = PlnPCA.from_formula("endog ~ 1", data=data, rank=5)
    >>> pca.fit()
    >>> print(pca)
    >>> pca.viz(colors=data["labels"])

    >>> from pyPLNmodels import PlnPCA, load_scrna
    >>> data = load_scrna()
    >>> pca = PlnPCA.from_formula("endog ~ 1 + labels", data=data, rank=5)
    >>> pca.fit()
    >>> print(pca)
    >>> pca.viz(colors=data["labels"])

    See also
    --------
    :class:`pyPLNmodels.PlnPCAcollection`
    :class:`pyPLNmodels.Pln`
    :class:`pyPLNmodels.ZIPlnPCA`
    """

    _components: torch.Tensor

    @_add_doc(
        BaseModel,
        params="""
            rank : int, optional(keyword-only)
                The rank of the approximation, by default 5.
            """,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA(data["endog"])
            >>> pca.fit()
            >>> print(pca)
        """,
        returns="""
            PlnPCA
        """,
        see_also="""
        :func:`pyPLNmodels.PlnPCA.from_formula`
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
        rank: int = 5,
    ):  # pylint: disable=too-many-arguments
        self._rank = rank
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
            rank : int, optional(keyword-only)
                The rank of the approximation, by default 5.
            """,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA.from_formula("endog ~ 1", data=data, rank=5)
        """,
        returns="""
            PlnPCA
        """,
        see_also="""
        :class:`pyPLNmodels.Pln`
        :func:`pyPLNmodels.PlnPCA.__init__`
    """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        rank: int = 5,
    ):
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
            rank=rank,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> plnpca = PlnPCA.from_formula("endog ~ 1", data)
        >>> plnpca.fit()
        >>> print(plnpca)

        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> plnpca = PlnPCA.from_formula("endog ~ 1", data)
        >>> plnpca.fit(maxiter=500, verbose=True)
        >>> print(plnpca)
        """,
        returns="""
        PlnPCA object
        """,
    )
    def fit(
        self,
        *,
        maxiter: int = 400,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL,
        verbose: bool = False,
    ):
        return super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    def _init_model_parameters(self):
        coef = _init_coef(endog=self._endog, exog=self._exog, offsets=self._offsets)
        if not hasattr(self, "_coef"):
            if coef is not None:
                self._coef = coef.detach().to(DEVICE)
            else:
                self._coef = None

        if not hasattr(self, "_components"):
            self._components = _init_components(self._endog, self.rank).to(DEVICE)

    def _init_latent_parameters(self):
        self._latent_mean = _init_latent_mean_pca(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            coef=self._coef,
            components=self._components,
        )
        if self.n_samples * self.rank**2 > 1e8:
            self._latent_sqrt_variance = (
                1 / 2 * torch.ones((self.n_samples, self.rank)).to(DEVICE)
            )
        else:
            self._latent_sqrt_variance = _init_latent_sqrt_variance_pca(
                marginal_mean=self._marginal_mean,
                offsets=self._offsets,
                components=self._components,
                mode=self._latent_mean,
            )

    @property
    def rank(self):
        """Rank of the covariance of the Gaussian latent variable."""
        return self._rank

    @property
    def _description(self):
        return f"{self.rank} principal components."

    @property
    @_add_doc(BaseModel)
    def list_of_parameters_needing_gradient(self):
        list_param = [self._latent_mean, self._latent_sqrt_variance, self._components]
        if self._coef is not None:
            return list_param + [self._coef]
        return list_param

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA.from_formula("endog ~ 1", data)
            >>> pca.fit()
            >>> elbo = pca.compute_elbo()
            >>> print(elbo)
        """,
    )
    def compute_elbo(self):
        return elbo_plnpca(
            endog=self._endog,
            marginal_mean=self._marginal_mean,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            components=self._components,
        )

    @property
    @_add_doc(BaseModel)
    def dict_model_parameters(self):
        return self._default_dict_model_parameters

    @property
    @_add_doc(BaseModel)
    def dict_latent_parameters(self):
        return self._default_dict_latent_parameters

    @property
    def components(self):
        """
        Returns the principal components of the PlnPCA model, i.e. the weights
        of features that explain the most variance in the data.

        Returns
        -------
        torch.Tensor
            The components with size (dim, rank)
        """
        return self._components.detach().cpu()

    @components.setter
    @_array2tensor
    def components(self, components: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the components.

        Parameters
        ----------
        components : torch.Tensor
            The components to set.

        Raises
        ------
        ValueError
            If the components have an invalid shape.
        """
        if components.shape != (self.dim, self.rank):
            raise ValueError(
                f"Wrong shape. Expected ({self.dim, self.rank}), got {components.shape}"
            )
        self._components = components

    @property  # Here only to be able to define a setter.
    @_add_doc(BaseModel)
    def coef(self):
        return super().coef

    @coef.setter
    @_array2tensor
    def coef(self, coef: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the `coef` property.

        Parameters
        ----------
        coef : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The coefficients of size (nb_cov, dim).

        Raises
        ------
        ValueError
            If the shape of the coef is incorrect.
        """
        if coef.shape != (self.nb_cov, self.dim):
            raise ValueError(
                f"Wrong shape for the coef. Expected ({(self.nb_cov, self.dim)}), got {coef.shape}"
            )
        self._coef = coef

    @property
    def _covariance(self):
        return self._components @ self._components.T

    def _get_two_dim_latent_variances(self, sklearn_components):
        transformed_components = sklearn_components @ self.components.numpy()
        transformed_components_latent_var = (
            transformed_components[np.newaxis, :]
            * self.latent_variance.numpy()[:, np.newaxis]
        )
        covariances = (
            transformed_components_latent_var @ transformed_components.T[np.newaxis, :]
        )
        return covariances

    @_add_doc(
        BaseModel,
        params="""
        project : bool, optional
            Whether to project the latent variables onto the `rank` first PCs, by default `False`.
        """,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA.from_formula("endog ~ 1", data)
            >>> pca.fit()
            >>> transformed_endog_low_dim = pca.transform()
            >>> transformed_endog_high_dim = pca.transform(project=False)
            >>> print(transformed_endog_low_dim.shape)
            >>> print(transformed_endog_high_dim.shape)
            >>> transformed_no_exog = pca.transform(remove_exog_effect=True, project=True)
        """,
    )
    def transform(self, remove_exog_effect: bool = False, project=False):
        if project is True:
            return self.projected_latent_variables(
                rank=self.rank, remove_exog_effect=remove_exog_effect
            )
        return super().transform(remove_exog_effect=remove_exog_effect)

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> pca = PlnPCA.from_formula("endog ~ 1", data)
        >>> pca.fit()
        >>> print(pca.latent_variables.shape)
        >>> pca.viz() # Visualize the latent variables
        """,
    )
    def latent_variables(self):
        return torch.matmul(self.latent_mean, self.components.T) + self.marginal_mean

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> pca = PlnPCA.from_formula("endog ~ 1", data)
        >>> pca.fit()
        >>> print(pca.latent_positions.shape)
        >>> pca.viz(remove_exog_effect=True) # Visualize the latent positions
        """,
    )
    def latent_positions(self):
        return torch.matmul(self.latent_mean, self.components.T)

    @property
    @_add_doc(BaseModel)
    def number_of_parameters(self):
        return self.dim * (self.nb_cov + self.rank) - self.rank * (self.rank - 1) / 2

    @property
    def _additional_attributes_list(self):
        return []

    @property
    def _additional_methods_list(self):
        return []

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> pca = PlnPCA.from_formula("endog ~ 1", data=data)
        >>> pca.fit()
        >>> pca.plot_correlation_circle(variable_names=["MALAT1", "ACTB"])
        >>> pca.plot_correlation_circle(variable_names=["A", "B"], indices_of_variables=[0, 4])
        """,
    )
    def plot_correlation_circle(
        self, variable_names, indices_of_variables=None, title: str = ""
    ):
        super().plot_correlation_circle(
            variable_names=variable_names,
            indices_of_variables=indices_of_variables,
            title=title,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> pca = PlnPCA.from_formula("endog ~ 1", data=data)
        >>> pca.fit()
        >>> pca.biplot(variable_names=["MALAT1", "ACTB"])
        >>> pca.biplot(variable_names=["A", "B"], indices_of_variables=[0, 4], colors=data["labels"])
        """,
    )
    def biplot(
        self,
        variable_names,
        *,
        indices_of_variables: np.ndarray = None,
        colors: np.ndarray = None,
        title: str = "",
    ):
        super().biplot(
            variable_names=variable_names,
            indices_of_variables=indices_of_variables,
            colors=colors,
            title=title,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> plnpca = PlnPCA.from_formula("endog ~ 1", data=data)
        >>> plnpca.fit()
        >>> plnpca.pca_pairplot(n_components=5)
        >>> plnpca.pca_pairplot(n_components=5, colors=data["labels"])
        """,
    )
    def pca_pairplot(self, n_components: bool = 3, colors=None):
        super().pca_pairplot(n_components=n_components, colors=colors)

    @property
    def _endog_predictions(self):
        covariance_a_posteriori = torch.sum(
            (self.components**2).unsqueeze(0)
            * (self.latent_sqrt_variance**2).unsqueeze(1),
            axis=2,
        )
        return torch.exp(
            self.offsets + self.latent_variables + 1 / 2 * covariance_a_posteriori
        )

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA(data["endog"])
            >>> pca.fit()
            >>> pca.plot_expected_vs_true()
            >>> pca.plot_expected_vs_true(colors=data["labels"])
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA.from_formula("endog ~ 1 + labels", data=data)
            >>> pca.fit()
            >>> pca.viz()
            >>> pca.viz(colors=data["labels"])
            >>> pca.viz(show_cov=True)
            >>> pca.viz(remove_exog_effect=True, colors=data["labels"])
            """,
    )
    def viz(
        self,
        *,
        ax=None,
        colors=None,
        show_cov: bool = False,
        remove_exog_effect: bool = False,
    ):
        super().viz(
            ax=ax,
            colors=colors,
            show_cov=show_cov,
            remove_exog_effect=remove_exog_effect,
        )
