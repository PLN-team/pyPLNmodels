import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class SAGA():
    '''Aims at computing the effective gradients of the SAGA algorithm.'''

    def __init__(self, params, sample_size):
        '''Define some usefuls attributes of the object, such as the parameters
        and the sample size. Need the sample size in order to initialize the
        table (this large vector will be needed to average the gradient.)

        Args:
            params: list. Each element of the list should be a torch.tensor object.
            sample_size: int. The number of sample in your dataset.
        Returns:
            None
        '''
        self.params = params
        self.sample_size = sample_size
        self.nb_non_zero = 0
        self.run_through = False
        self.bias = 1
        for param in params:
            shape = list(param.shape)
            shape.insert(0, sample_size)
            # Initialization of the table for each param with zeros.
            param.table = torch.zeros(shape).to(device)
            param.mean_table = torch.zeros(param.shape).to(device)

    def computeAndSetVarianceReducedGrad(self, batch_grads, selected_indices):
        '''Update the gradient of each parameter of the object with the SAGA formula.
        Note that it only needs to change the bias to get the SAG formula.

        Args:
            batch_grads: list of torch.tensors objects. Each object should be of size
                (batch_size, parameter.shape). Note that the input list should match
                the input list in the initialization. i.e., if the list in the
                __init__ begins with the parameter beta, this list should begins with
                the corresponding gradients for parameter beta.
            selected_indices: list of size batch_size. The indices of your batch. We
                need this to store the new gradients in the table. If Y is your dataset,
                then Y_batch should be equal to Y[selected_indices].
        Returns:
            None but updates the gradient of each parameter with the variance reducted
            gradient.
        '''
        means_batch_table = []
        self.batch_size = len(selected_indices)
        # Number of samples already seen.
        self.nb_non_zero = min(
            self.nb_non_zero +
            self.batch_size,
            self.sample_size)

        for i, param in enumerate(self.params):
            means_batch_table = torch.mean(
                param.table[selected_indices], axis=0)
            # Gradient formula in the SAGA optimizer
            batch_grad = torch.mean(batch_grads[i], axis=0)
            param.grad = (self.bias * (batch_grad -
                          means_batch_table) + param.mean_table)
            # Update the table with the new gradients
            if self.run_through == False:
                param.mean_table *= (self.nb_non_zero -
                                     self.batch_size) / (self.nb_non_zero)
                param.mean_table += (self.batch_size /
                                     (self.nb_non_zero) *
                                     (batch_grad)).detach()
            else:
                param.mean_table -= ((self.batch_size / self.sample_size)
                                     * (means_batch_table - batch_grad)).detach()
            # UPDATE OF THE TABLE
            param.table[selected_indices] = batch_grads[i].detach()
        if self.nb_non_zero == self.sample_size:
            self.run_through = True

