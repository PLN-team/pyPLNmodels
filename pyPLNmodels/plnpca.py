from typing import Optional, Union
import torch
import numpy as np
import pandas as pd

from pyPLNmodels.base import BaseModel
from pyPLNmodels.elbos import elbo_plnpca
from pyPLNmodels._initialization import _init_coef, _init_components, _init_latent_mean
from pyPLNmodels._data_handler import _extract_data_from_formula
from pyPLNmodels._utils import _add_doc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnPCA(BaseModel):
    """Principal Component Analysis on top of a PLN model, that is a PLN
    model with low rank covariance, adapted to datasets with lots of features.
    For more details, see  Chiquet, J., Mariadassou, M., Robin, S.
    “Variational inference for probabilistic Poisson PCA.” Annals of applied stats.

    Examples
    --------
    >>> from pyPLNmodels import PlnPCA, load_scrna, get_simulation_parameters, sample_pln
    >>> endog, labels = load_scrna(return_labels = True, for_formula = False)
    >>> data = {"endog": endog}
    >>> pca = PlnPCA.from_formula("endog ~ 1", data = data, rank = 5)
    >>> pca.fit()
    >>> print(pca)
    >>> pca.viz(colors = labels)

    >>> plnparam = get_simulation_parameters(n_samples =100, dim = 60, nb_cov = 2, rank = 8)
    >>> endog = sample_pln(plnparam)
    >>> data = {"endog": endog, "cov": plnparam.exog, "offsets": plnparam.offsets}
    >>> plnpca = PlnPCA.from_formula("endog ~ 0 + cov", data = data, rank = 5)
    >>> plnpca.fit()
    >>> print(plnpca)

    See also
    --------
    :class:`pyPLNmodels.PlnPCAcollection`
    :class:`pyPLNmodels.Pln`
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
            >>> pca = PlnPCA.from_formula("endog ~ 1", data)
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
            >>> pca = PlnPCA.from_formula("endog ~ 1", data = data, rank = 5)
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

    def _init_model_parameters(self):
        coef = _init_coef(endog=self._endog, exog=self._exog, offsets=self._offsets)
        if coef is not None:
            self._coef = coef.detach().to(DEVICE)
        else:
            self._coef = None
        self._components = _init_components(self._endog, self.rank).to(DEVICE)

    def _init_latent_parameters(self):
        self._latent_mean = _init_latent_mean(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            coef=self._coef,
            components=self._components,
        )
        self._latent_sqrt_variance = (
            1 / 2 * torch.ones((self.n_samples, self.rank)).to(DEVICE)
        )

    @property
    def rank(self):
        """Rank of the covariance of the gaussian latent variable."""
        return self._rank

    @property
    def _description(self):
        return f"{self.rank} principal components."

    @property
    @_add_doc(BaseModel)
    def list_of_parameters_needing_gradient(self):
        list_no_coef = [self._latent_mean, self._latent_sqrt_variance, self._components]
        if self._coef is not None:
            return list_no_coef + [self._coef]
        return list_no_coef

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
            self._endog,
            self._marginal_mean,
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_variance,
            self._components,
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
        of features that explains the most variance in the data.

        Returns
        -------
        torch.Tensor
            The components with size (dim, rank)
        """
        return self._components.detach().cpu()

    @property
    def _covariance(self):
        return self._components @ self._components.T

    def _get_two_dim_covariances(self, sklearn_components):
        transformed_components = sklearn_components @ self.components.numpy()
        transformed_components_latent_var = (
            transformed_components[np.newaxis, :]
            * self.latent_variance.numpy()[:, np.newaxis]
        )
        covariances = (
            transformed_components_latent_var @ transformed_components.T[np.newaxis, :]
        )
        return covariances

    @_add_doc(BaseModel)
    def transform(self, project=False):
        """Transforms the data"""
        if project is True:
            return self.projected_latent_variables
        return self.latent_variables

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> pca = PlnPCA.from_formula("endog ~ 1", data)
        >>> pca.fit()
        >>> print(pca.latent_variables.shape)
        """,
    )
    def latent_variables(self):
        return torch.matmul(self.latent_mean, self.components.T) + self.marginal_mean

    @property
    @_add_doc(BaseModel)
    def number_of_parameters(self):
        return self.dim * (self.nb_cov + self.rank) - self.rank * (self.rank - 1) / 2

    @property
    def _additional_properties_list(self):
        return ""

    @property
    def _additional_methods_list(self):
        return ""

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> plnpca = PlnPCA.from_formula("endog ~ 1", data = data)
        >>> plnpca.fit()
        >>> plnpca.plot_correlation_circle(["A","B"], indices_of_variables = [4,8])
        >>> should add some plot with pd.DataFrame.
        """,
    )
    def plot_correlation_circle(
        self, variables_names, indices_of_variables=None, title: str = ""
    ):
        super().plot_correlation_circle(
            variables_names=variables_names,
            indices_of_variables=indices_of_variables,
            title=title,
        )

    def _get_max_n_components(self):
        return self.rank

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> plnpca = PlnPCA.from_formula("endog ~ 1", data = data)
        >>> plnpca.fit()
        >>> plnpca.pca_pairplot(n_components = 5)
        """,
    )
    def pca_pairplot(self, n_components=None, colors=None):
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
