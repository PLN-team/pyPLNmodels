import torch


class SandwichPln:  # pylint: disable=too-many-instance-attributes
    """
    Implements the variance estimation of the coefficients of a PLN model.

    The computations are based on "Evaluating Parameter Uncertainty in the Poisson Lognormal
    Model with Corrected Variational Estimators" from Batardière, B., Chiquet, J., Mariadassou, M.

    See also
    --------
    :class:`pyPLNmodels.Pln`
    """

    def __init__(self, pln):
        """
        Instantiate all the relevant values.
        """
        self._endog = pln._endog
        self._exog = pln._exog
        self.nb_cov = pln.nb_cov
        self.dim = pln.dim
        self.n_samples = pln.n_samples
        self._latent_sqrt_variance = pln._latent_sqrt_variance.detach()
        self._latent_mean = pln._latent_mean.detach()
        self._offsets = pln._offsets
        if len(pln._covariance.shape) == 1:
            self._covariance = torch.diag(pln._covariance).detach()
        else:
            self._covariance = pln._covariance.detach()
        self._inv_covariance = torch.inverse(self._covariance)
        self._pred_endog = torch.exp(
            self._offsets + self._latent_mean + 0.5 * self._latent_sqrt_variance**2
        ).detach()

    def get_mat_dn(self):
        """
        Gets the `D_n` matrix of the sandwich estimator. For more details, see "Evaluating Parameter
        Uncertainty in the Poisson Lognormal Model with Corrected Variational
        Estimators" from Batardière, B., Chiquet, J., Mariadassou, M.
        """
        endog_minus_pred = self._endog - self._pred_endog
        endog_outer = torch.matmul(
            endog_minus_pred.unsqueeze(2), endog_minus_pred.unsqueeze(1)
        )
        exog_outer = torch.matmul(self._exog.unsqueeze(2), self._exog.unsqueeze(1))
        res = torch.zeros(self.nb_cov * self.dim, self.nb_cov * self.dim).to(
            self._endog.device
        )
        for i in range(self.n_samples):
            res += torch.kron(endog_outer[i], exog_outer[i])
        return res / self.n_samples

    def _get_mat_i_cn(self, i):
        a_i = self._pred_endog[i]
        s_i = self._latent_sqrt_variance[i]
        diag_mat_i = torch.diag(
            1 / a_i + s_i**4 / (1 + s_i**2 * (a_i + torch.diag(self._inv_covariance)))
        )
        return torch.inverse(self._covariance + diag_mat_i)

    def get_mat_cn(self):
        """
        Gets the `C_n` matrix of the sandwich estimator. For more details, see "Evaluating Parameter
        Uncertainty in the Poisson Lognormal Model with Corrected Variational
        Estimators" from Batardière, B., Chiquet, J., Mariadassou, M.
        """
        exog_outer = torch.clone(
            torch.matmul(self._exog.unsqueeze(2), self._exog.unsqueeze(1))
        ).to(self._endog.device)
        mat_cn = torch.zeros(self.nb_cov * self.dim, self.nb_cov * self.dim).to(
            self._endog.device
        )
        for i in range(self.n_samples):
            mat_i = self._get_mat_i_cn(i)
            big_mat = torch.zeros(mat_i.shape).to(self._endog.device)
            big_mat[:] = mat_i[:]
            exog_i = exog_outer[i].clone().detach()
            mat_cn += torch.kron(big_mat, exog_i)
        return -mat_cn / self.n_samples

    def get_sandwich(self):
        """
        Gets the sandwich (matrix) estimator. For more details,
        see "Evaluating Parameter Uncertainty in the Poisson Lognormal Model
        with Corrected Variational Estimators" from Batardière, B., Chiquet, J., Mariadassou, M.
        """
        mat_dn = self.get_mat_dn()
        mat_cn = self.get_mat_cn()
        inv_mat_cn = torch.inverse(mat_cn)
        return torch.mm(torch.mm(inv_mat_cn, mat_dn), inv_mat_cn)

    def get_variance_coef(self):
        """
        Gets the variance of the estimator of the coefficients. For more details,
        see "Evaluating Parameter Uncertainty in the Poisson Lognormal Model
        with Corrected Variational Estimators" from Batardière, B., Chiquet, J., Mariadassou, M.
        """
        sandwich = self.get_sandwich()
        diag_sandwich = torch.diag(sandwich)
        return (diag_sandwich / self.n_samples).reshape(self.nb_cov, self.dim)
