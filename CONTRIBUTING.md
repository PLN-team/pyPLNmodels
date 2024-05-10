# What to work on

A public roadmap will be available soon.


## Fork/clone/pull

The typical workflow for contributing is:

1. Fork the `main` branch from the [GitLab repository](https://forgemia.inra.fr/bbatardiere/pyplnmodels).
2. Clone your fork locally.
3. Run `pip install pre-commit` if pre-commit is not already installed.
4. Inside the repository, run '''pre-commit install'''.
5. Commit changes.
6. Push the changes to your fork.
7. Send a pull request from your fork back to the original `main` branch.

## How to implement a new model
You can implement a new model `newmodel` by inheriting from the abstract `_model` class in the `models` module.
The `newmodel` class should contains at least the following code:
```
class newmodel(_model):
    _NAME=""
    @property
    def latent_variables(self) -> torch.Tensor:
        "Implement here"

    def compute_elbo(self) -> torch.Tensor:
        "Implement here"

    def _compute_elbo_b(self) -> torch.Tensor:
        "Implement here"

    def _smart_init_model_parameters(self)-> None:
        "Implement here"

    def _random_init_model_parameters(self)-> None:
        "Implement here"

    def _smart_init_latent_parameters(self)-> None:
        "Implement here"

    def _random_init_latent_parameters(self)-> None:
        "Implement here"

    @property
    def _list_of_parameters_needing_gradient(self)-> list:
        "Implement here"
    @property
    def _description(self)-> str:
        "Implement here"

    @property
    def number_of_parameters(self) -> int:
        "Implement here"

    @property
    def model_parameters(self)-> Dict[str, torch.Tensor]:
        "Implement here"

    @property
    def latent_parameters(self)-> Dict[str, torch.Tensor]:
        "Implement here"
```
Each value of the 'latent_parameters' dict should be implemented (and protected) both in the
`_random_init_latent_parameters` and '_smart_init_latent_parameters'.
Each value of the 'model_parameters' dict should be implemented (and protected) both in the
`_random_init_model_parameters` and '_smart_init_model_parameters'.
For example, if you have one model parameters `coef` and latent_parameters `latent_mean` and `latent_var`, you should implement such as
```py
class newmodel(_model):
    @property
    def model_parameters(self) -> Dict[str, torch.Tensor]:
        return {"coef":self.coef}
    @property
    def latent_parameters(self) -> Dict[str, torch.Tensor]:
        return {"latent_mean":self.latent_mean, "latent_var":self.latent_var}

    def _random_init_latent_parameters(self):
        self._latent_mean = init_latent_mean()
        self._latent_var = init_latent_var()

    @property
    def _smart_init_model_parameters(self):
        self._latent_mean = random_init_latent_mean()
        self._latent_var = random_init_latent_var()

    @property
    def latent_var(self):
        return self._latent_var

    @property
    def latent_mean(self):
        return self._latent_mean

    def _random_init_model_parameters(self):
        self._coef = init_coef()

    def _smart_init_model_parameters(self):
        self._coef = random_init_latent_coef()

    @property
    def coef(self):
        return self._coef
```



Then, add `newmodel` in the `__init__.py` file of the pyPLNmodels module.
If `newmodel` is well implemented, running
```
from pyPLNmodels import newmodel, get_real_count_data

endog = get_real_count_data()
model = newmodel(endog, add_const = True)
model.fit(nb_max_iteration = 10, tol = 0)
```
should increase the elbo of the model. You should document your functions with
[numpy-style
docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). You can use
the `_add_doc` decorator (implemented in the `_utils` module) to inherit the docstrings of the `_model` class.
