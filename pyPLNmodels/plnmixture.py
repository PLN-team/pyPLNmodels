from typing import Optional, Union
import torch
import pandas as pd
import numpy as np

from pyPLNmodels.base import BaseModel, DEFAULT_TOL
from pyPLNmodels._data_handler import _extract_data_from_formula
from pyPLNmodels._utils import _add_doc
from pyPLNmodels._initialization import _init_gmm
from pyPLNmodels.elbos import (
    weighted_elbo_pln_diag,
    persample_elbo_pln_diag,
    per_sample_elbo_pln_mixture_diag,
)
from pyPLNmodels.plndiag import PlnDiag


import seaborn as sns
import matplotlib.pyplot as plt


class PlnMixture(BaseModel):
    """
    Pln mixture models, that is a gaussian mixture model with Poisson layer on top of it.

    Examples
    --------
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> from pyPLNmodels import PlnMixture, load_scrna
    >>> data = load_scrna()
    >>> mixture = PlnMixture(data["endog]",n_clusters = 2)
    >>> mixture.fit()
    >>> print(mixture)
    >>> pair_confusion_matrix(mixture.clusters, data["labels"])
    >>> sns.show()


    See also
    --------
    :func:`pyPLNmodels.PlnDiag`
    :func:`pyPLNmodels.Pln`
    :func:`pyPLNmodels.PlnMixture.__init__`
    :func:`pyPLNmodels.PlnMixture.from_formula`
    :func:`pyPLNmodels.PlnMixture.predict
    """

    @_add_doc(
        BaseModel,
        params="""
            n_clusters : int
                The number of clusters in the model.
            """,
        example="""
            >>> from pyPLNmodels import PlnMixture, load_scrna
            >>> data = load_scrna()
            >>> mixture = PlnMixture(data["endog]",n_clusters = 2)
            >>> mixture.fit()
            >>> print(mixture)
        """,
        returns="""
            PlnMixture
        """,
        see_also="""
        :func:`pyPLNmodels.PlnMixture.from_formula`
        """,
    )
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        n_clusters: int,
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):  # pylint: disable=too-many-arguments
        self._n_clusters = n_clusters
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
            n_clusters : int
                The number of clusters in the model.
            """,
        example="""
            >>> from pyPLNmodels import PlnMixture, load_scrna
            >>> data = load_scrna()
            >>> mixture = PlnMixture.from_formula("endog ~ 1",data, n_clusters = 2)
            >>> mixture.fit()
            >>> print(mixture)
        """,
        returns="""
            PlnMixture
        """,
        see_also="""
        :func:`pyPLNmodels.PlnMixture`
        """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        n_clusters: int,
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
            n_clusters=n_clusters,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 1", data, n_clusters = 2)
        >>> mixture.fit()
        >>> print(mixture)

        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 1", data, n_clusters = 2)
        >>> mixture.fit(maxiter=500, verbose=True)
        >>> print(mixture)
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
        super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    @property
    def _additional_attributes_list(self):
        return [
            ".clusters",
            ".prob_clusters",
            ".covariances",
            ".coefs",
            "cluster_bias",
            "latent_prob",
        ]

    @property
    def _additional_methods_list(self):
        return []

    @property
    def _covariance(self):
        raise NotImplementedError("No single covariance. Call ._covariances.")

    @property
    def _covariances(self):
        return self._sqrt_covariances**2

    @property
    def covariances(self):
        return self._covariances.detach().cpu()

    @property
    def weights(self):
        return self._weights.cpu()

    @property
    def covariance(self):
        """The GMM does not have a single covariance. It has multiple covariances, one per cluster.
        You may call .covariances."""
        raise NotImplementedError(
            "The GMM does not have a single covariance. It has multiple covariances, one"
            " per cluster. You may call .covariances."
        )

    def _init_parameters(self):
        pln = PlnDiag(
            endog=self._endog, exog=self._exog, offsets=self._offsets, add_const=True
        )
        pln.fit(maxiter=150)
        self._cluster_bias, self._sqrt_covariances, self._weights, self._latent_prob = (
            _init_gmm(pln.latent_positions, self.n_clusters)
        )
        if pln.nb_cov > 1:
            self._coef = pln.coef[:-1]  # Should not retrieve the mean
        else:
            self._coef = None
        self._latent_means = self._cluster_bias.unsqueeze(1)
        if self._exog is not None:
            self._latent_means = self._latent_means + self._marginal_mean.unsqueeze(0)
        else:
            self._latent_means = self._latent_means.repeat_interleave(
                self.n_samples, dim=1
            )

        self._latent_sqrt_variances = torch.randn(
            self.n_clusters, self.n_samples, self.dim
        )

    def _init_latent_parameters(self):
        """Everything is done in the _init_parameters method."""

    def _init_model_parameters(self):
        """Everything is done in the _init_parameters method."""

    @property
    def _description(self):
        return f"diagonal covariances and {self.n_clusters} clusters."

    @property
    def _endog_predictions(self):
        pass

    def _get_two_dim_covariances(self):
        pass

    def biplot(self):
        pass

    def compute_elbo(self):
        if self._exog is None:
            marginal_means = self._cluster_bias.unsqueeze(1).repeat_interleave(
                self.n_samples, dim=1
            )
        else:
            marginal_means = self._marginal_mean.unsqueeze(
                0
            ) + self._cluster_bias.unsqueeze(1)
        self.per_sample_per_cluster_elbo = per_sample_elbo_pln_mixture_diag(
            endog=self._endog,
            marginal_means=marginal_means,
            offsets=self._offsets,
            latent_means=self._latent_means,
            latent_sqrt_variances=self._latent_sqrt_variances,
            diag_precisions=1 / (self._covariances),
        )
        elbo = torch.sum(self.per_sample_per_cluster_elbo * self._latent_prob.T)
        entropy = torch.sum(torch.xlogy(self._latent_prob, self._weights.unsqueeze(0)))
        entropy -= torch.sum(torch.xlogy(self._latent_prob, self._latent_prob))
        return elbo + entropy

    def _update_closed_forms(self):
        log_latent_prob = (
            torch.log(self._weights.unsqueeze(0)) + self.per_sample_per_cluster_elbo.T
        )
        with torch.no_grad():
            self._latent_prob = torch.nn.Softmax(dim=1)(log_latent_prob)
            self._weights = torch.mean(self._latent_prob, axis=0)

    @property
    def dict_latent_parameters(self):
        """Dictionary of latent parameters."""
        return {
            "latent_prob": self.latent_prob,
            "latent_means": self.latent_means,
            "latent_sqrt_variances": self.latent_sqrt_variances,
            "cluster_bias": self.cluster_bias,
        }

    @property
    def dict_model_parameters(self):
        """Dictionary of model parameters."""
        return {
            "coef": self.coef,
            "covariances": self.covariances,
            "weights": self.weights,
        }

    @property
    def latent_positions(self):
        """Latent positions."""

    @property
    def latent_variables(self):
        """Latent variables."""

    @property
    def latent_means(self):
        """The latent mean for each cluster."""
        return self._latent_means.detach().cpu()

    @property
    def latent_prob(self):
        return self._latent_prob.detach().cpu()

    @property
    def list_of_parameters_needing_gradient(self):
        """List of parameters needing gradient."""
        list_param = [
            self._latent_means,
            self._latent_sqrt_variances,
            self._sqrt_covariances,
            self._cluster_bias,
        ]
        if self._exog is not None:
            list_param.append(self._coef)
        return list_param

    @property
    def number_of_parameters(self):
        """Number of parameters."""

    def pca_pairplot(self):
        """PCA pair plot."""

    def plot_correlation_circle(self):
        """Plot correlation circle."""

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def cluster_bias(self):
        return self._cluster_bias.detach().cpu()

    def predict_clusters(self):
        return torch.argmax(self._latent_prob, axis=1)
