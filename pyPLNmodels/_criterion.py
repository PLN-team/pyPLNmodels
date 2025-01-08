import numpy as np

BETA = 0.03


class _LossCriterionMonitor:
    """
    Class that deals with the criterion of PLN models, based on a
    moving average of the hessian of the slope of the cumulative elbo.
    """

    def __init__(self):
        """
        Initialize the LossCriterionMonitor class.
        """
        self.running_times = []
        self.elbo_list = []
        self.new_derivative = 0
        self.criterion = 1
        self.current_hessian = 0

    def update_criterion(self, elbo, running_time):
        """
        Update the moving average hessian criterion based on the elbo.
        """
        self._update_lists(elbo, running_time)
        if self._iteration_number > 1:
            self._update_derivative()
            self._update_criterion()

    def _update_lists(self, elbo, running_time):
        self.elbo_list.append(elbo)
        self.running_times.append(running_time)

    def _update_derivative(self):
        normalized_elbo_list = [
            -elbo / self._cumulative_elbo for elbo in self.elbo_list
        ]
        current_derivative = np.abs(normalized_elbo_list[-2] - normalized_elbo_list[-1])
        old_derivative = self.new_derivative
        self.new_derivative = (
            self.new_derivative * (1 - BETA) + current_derivative * BETA
        )
        self.current_hessian = np.abs(self.new_derivative - old_derivative)

    def _update_criterion(self):
        self.criterion = self.criterion * (1 - BETA) + self.current_hessian * BETA

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
        return sum(self.elbo_list)
