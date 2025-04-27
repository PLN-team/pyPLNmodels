from typing import Union, Optional

import pandas as pd
import torch
import numpy as np

from pyPLNmodels.models.pln import Pln
from pyPLNmodels.models.base import BaseModel
from pyPLNmodels.calculations._closed_forms import _closed_formula_diag_covariance
from pyPLNmodels.calculations.elbos import profiled_elbo_pln_diag
from pyPLNmodels.calculations.entropies import entropy_gaussian
from pyPLNmodels.utils._utils import _add_doc, _shouldbefitted, _none_if_no_exog
from pyPLNmodels.utils._viz import DiagModelViz


class PlnDiag(Pln):
    """
    PLN model with diagonal covariance. Inference is faster, as well as memory requirements.

    Examples
    --------
    >>> from pyPLNmodels import PlnDiag, load_scrna
    >>> data = load_scrna()
    >>> pln = PlnDiag(data["endog"])
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors=data["labels"])

    >>> from pyPLNmodels import PlnDiag, load_scrna
    >>> data = load_scrna()
    >>> pln = PlnDiag.from_formula("endog ~ 1 + labels", data=data)
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors=data["labels"])
    """

    _ModelViz = DiagModelViz

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln.from_formula("endog ~ 1", data)
            >>> pln.fit()
            >>> print(pln)
        """,
        returns="""
            Pln
        """,
        see_also="""
        :func:`pyPLNmodels.Pln.from_formula`
        :class:`pyPLNmodels.PlnPCA`
        :class:`pyPLNmodels.PlnMixture`
        """,
    )
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )

    @property
    def _covariance(self):
        return _closed_formula_diag_covariance(
            marginal_mean=self._marginal_mean,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            n_samples=self.n_samples,
        )

    @property
    def _precision(self):
        return 1 / self._covariance

    def compute_elbo(self):
        return profiled_elbo_pln_diag(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
        )

    @_none_if_no_exog
    @_shouldbefitted
    def get_variance_coef(self):
        """
        Calculate the variance of the regression coefficients using the sandwich estimator.
        Returns None if there are no exogenous variables in the model.

        Returns
        -------
        torch.Tensor
            Variance of the regression coefficients.

        Raises
        ------
        ValueError
            If the number of samples is less than the product of the
            number of covariates and dimensions.

        Examples
        --------
        >>> from pyPLNmodels import PlnDiag, load_scrna
        >>> rna_data = load_scrna(dim = 10)
        >>> pln = PlnDiag(rna_data["endog"], exog=rna_data["labels_1hot"], add_const=False)
        >>> pln.fit()
        >>> variance = pln.get_variance_coef()
        >>> print('variance', variance)

        See also
        --------
        :func:`pyPLNmodels.PlnDiag.summary`
        :func:`pyPLNmodels.PlnDiag.get_coef_p_values`
        :func:`pyPLNmodels.PlnDiag.get_confidence_interval_coef`
        """
        return super().get_variance_coef()

    @_none_if_no_exog
    @_shouldbefitted
    def get_confidence_interval_coef(self, alpha: float = 0.05):
        """
        Calculate the confidence intervals for the regression coefficients.
        Returns None if there are no exogenous variables in the model.

        Parameters
        ----------
        alpha : float (optional)
            Significance level for the confidence intervals. Defaults to 0.05.

        Returns
        -------
        interval_low, interval_high : Tuple(torch.Tensor, torch.Tensor)
            Lower and upper bounds of the confidence intervals for the coefficients.

        Examples
        --------
        >>> from pyPLNmodels import PlnDiag, load_scrna
        >>> rna_data = load_scrna(dim = 10)
        >>> pln = PlnDiag(rna_data["endog"], exog=rna_data["labels_1hot"], add_const=False)
        >>> pln.fit()
        >>> interval_low, interval_high = pln.get_confidence_interval_coef()

        >>> import torch
        >>> from pyPLNmodels import PlnDiag, PlnDiagSampler
        >>>
        >>> sampler = PlnDiagSampler(n_samples=1500, add_const=False, nb_cov=4)
        >>> endog = sampler.sample() # Sample PlnDiag data.
        >>>
        >>> pln = PlnDiag(endog, exog=sampler.exog, add_const=False)
        >>> pln.fit()
        >>> interval_low, interval_high = pln.get_confidence_interval_coef(alpha=0.05)
        >>> true_coef = sampler.coef
        >>> inside_interval = (true_coef > interval_low) & (true_coef < interval_high)
        >>> print('Should be around 0.95:', torch.mean(inside_interval.float()).item())

        See also
        --------
        :func:`pyPLNmodels.PlnDiag.summary`
        :func:`pyPLNmodels.PlnDiag.get_coef_p_values`
        """
        return super().get_confidence_interval_coef(alpha=alpha)

    @_none_if_no_exog
    @_shouldbefitted
    def get_coef_p_values(self):
        """
        Calculate the p-values for the regression coefficients.
        Returns None if there are no exogenous variables in the model.

        Returns
        -------
        p_values : torch.Tensor
            P-values for the regression coefficients.

        Examples
        --------
        >>> from pyPLNmodels import PlnDiag, load_scrna
        >>> rna_data = load_scrna(dim = 10)
        >>> pln = PlnDiag(rna_data["endog"], exog=rna_data["labels_1hot"], add_const=False)
        >>> pln.fit()
        >>> p_values = pln.get_coef_p_values()
        >>> print('P-values: ', p_values)

        See also
        --------
        :func:`pyPLNmodels.PlnDiag.summary`
        :func:`pyPLNmodels.PlnDiag.get_confidence_interval_coef`
        """
        return super().get_coef_p_values()

    @_none_if_no_exog
    @_shouldbefitted
    def summary(self):
        """
        Print a summary of the regression coefficients and their p-values for each dimension.
        Returns None if there are no exogenous variabes in the model.

        Examples
        --------
        >>> from pyPLNmodels import PlnDiag, load_scrna
        >>> rna_data = load_scrna(dim = 10)
        >>> pln = PlnDiag(rna_data["endog"], exog = rna_data["labels_1hot"], add_const = False)
        >>> pln.fit()
        >>> pln.summary()

        See also
        --------
        :func:`pyPLNmodels.PlnDiag.get_confidence_interval_coef`
        """
        return super().summary()

    @property
    @_add_doc(BaseModel)
    def number_of_parameters(self):
        return self.dim * (self.nb_cov + 1)

    @property
    @_add_doc(BaseModel)
    def entropy(self):
        return entropy_gaussian(self.latent_variance).item()
