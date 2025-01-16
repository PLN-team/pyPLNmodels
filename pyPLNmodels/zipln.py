# pylint: skip-file
import torch

from pyPLNmodels import BaseModel


class ZIPln(BaseModel):
    """
    Zero-Inflated Pln (ZIPln) class. Like a Pln but adds zero-inflation
    Fitting such a model is slower than fitting a Pln.

    Examples
    --------
    >>> from pyPLNmodels import ZIPln, Pln, load_microcosm
    >>> data = load_microcosm() # microcosm are higly zero-inflated (96% of zeros)
    >>> zi = ZIPln.from_formula("endog ~ 1 + site", data)
    >>> zi.fit()
    >>> zi.viz(colors = data["site"])
    >>> # Here Pln is not appropriate:
    >>> pln = Pln.from_formula("endog ~ 1 + site", data)
    >>> pln.fit()
    >>> pln.viz(colors = data["site"])
    >>> # Can also give different covariates:
    >>> zi_diff = ZIPln.from_formula("endog ~ 1 + site | 1 + time", data)
    >>> zi.fit()
    >>> zi.viz(colors = data["site"])
    >>> ## Or take all the covariates
    >>> zi_all = ZIPln.from_formula("endog ~ 1 + site*time | 1 + site*time", data)
    >>> zi_all.fit()
    """

    _latent_prob: torch.Tensor
    _coef_inflation: torch.Tensor
    _dirac: torch.Tensor

    def __init__(
        self,
        endog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        exog_inflation: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        add_const: bool = True,
        add_const_inflation: bool = True,
    ):
        """
        Initializes the ZIPln class.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The count data.
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data. Defaults to None.
        exog_inflation : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data for the inflation part. Defaults to None. If None,
            will automatically add a vector of one if zero_inflation_formula is
            either "column-wise" or "row-wise".
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The offsets data. Defaults to None.
        add_const : bool, optional(keyword-only)
            Whether to add a column of one in the exog. Defaults to True.
        add_const_inflation : bool, optional(keyword-only)
            Whether to add a column of one in the exog_inflation. Defaults to True.
            If exog_inflation is None and zero_inflation_formula is not "global",
            add_const_inflation is set to True anyway and a warnings
            is launched.
        Returns
        -------
        A ZIPln object
        See also
        --------
        :func:`pyPLNmodels.ZIPln.from_formula`
        Examples
        --------
        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> rna = load_scrna()
        >>> zi = ZIPln(rna["endog"], add_const = True)
        >>> zi.fit()
        >>> print(zi)
        """
        (
            self._endog,
            self._exog,
            self._exog_inflation,
            self._offsets,
            self.column_endog,
            self._dirac,
            self._batch_size,
            self.samples_only_zeros,
        ) = _handle_data_with_inflation(
            endog,
            exog,
            exog_inflation,
            offsets,
            offsets_formula,
            self._zero_inflation_formula,
            take_log_offsets,
            add_const,
            add_const_inflation,
            batch_size,
        )
        self._fitted = False
        self._criterion_args = _CriterionArgs()
        if dict_initialization is not None:
            self._set_init_parameters(dict_initialization)
