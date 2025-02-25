from typing import Optional, Union

from pyPLNmodels import Pln

from pyPLNmodels._data_handler import _format_clusters, _check_dimensions_equal


import torch
import pandas as pd
import numpy as np


class PlnLDA(Pln):
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        clusters: Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
    ):  # pylint: disable=too-many-arguments
        self._clusters = _format_clusters(clusters)
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )
        _check_dimensions_equal(
            "endog", "clusters", self.n_samples, self._clusters.shape[0], 0, 0
        )
        self._exog_and_clusters = torch.cat((self.exog, self._clusters), dim=1)

    @property
    def _marginal_mean_clusters(self):
        pass
