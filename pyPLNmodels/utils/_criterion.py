import numpy as np

BETA = 0.03


class _ElboCriterionMonitor:
    """
    Class that deals with the criterion of PLN models, based on a
    moving average of the Hessian of the slope of the cumulative ELBO.
    """

    def __init__(self):
        """
        Initialize the `LossCriterionMonitor` class.
        """
        self.elbo_list = []
        self.cumulative_elbo_list = [0]
        self.normalized_elbo_list = []
        self.new_derivative = 0
        self.criterion_list = [1]
        self.current_hessian = 0

    def update_criterion(self, elbo):
        """
        Update the moving average Hessian criterion based on the ELBO.
        """
        self.elbo_list.append(elbo)
        self.cumulative_elbo_list.append(elbo + self._cumulative_elbo)
        self.normalized_elbo_list.append(elbo / self._cumulative_elbo)
        if self.iteration_number > 1:
            self._update_derivative()
            self._update_criterion()

    def _update_derivative(self):
        current_derivative = np.abs(
            self.normalized_elbo_list[-2] - self.normalized_elbo_list[-1]
        )
        old_derivative = self.new_derivative
        self.new_derivative = (
            self.new_derivative * (1 - BETA) + current_derivative * BETA
        )
        self.current_hessian = np.abs(self.new_derivative - old_derivative)

    def _update_criterion(self):
        new_criterion = self.criterion * (1 - BETA) + self.current_hessian * BETA
        self.criterion_list.append(new_criterion.item())

    @property
    def iteration_number(self) -> int:
        """
        Number of iterations done when fitting the model.

        Returns
        -------
        int
            The number of iterations.
        """
        return len(self.elbo_list)

    @property
    def _cumulative_elbo(self):
        return self.cumulative_elbo_list[-1]

    @property
    def criterion(self):
        """The current criterion of the associated model."""
        return self.criterion_list[-1]
