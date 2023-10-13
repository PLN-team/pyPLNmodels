# What to work on

A public roadmap will be available soon.


## Fork/clone/pull

The typical workflow for contributing is:

1. Fork the `main` branch from the [GitLab repository](https://forgemia.inra.fr/bbatardiere/pyplnmodels).
2. Clone your fork locally.
3. Run `pip install pre-commit` if pre-commit is not already installed.
4. Inside the repository, run 'pre-commit install'.
5. Commit changes.
6. Push the changes to your fork.
7. Send a pull request from your fork back to the original `main` branch.

## How to implement a new model
You can implement a new model `newmodel` by inheriting from the abstract `_model` class in the `models` module.
The `newmodel` class should contains at least the following code:
```
class newmodel(_model):
    _NAME=""
    def _random_init_latent_sqrt_var(self):
        "Implement here"

    @property
    def latent_variables(self):
        "Implement here"

    def compute_elbo(self):
        "Implement here"

    def _compute_elbo_b(self):
        "Implement here"

    def _smart_init_model_parameters(self):
        "Implement here"

    def _random_init_model_parameters(self):
        "Implement here"

    def _smart_init_latent_parameters(self):
        "Implement here"

    def _random_init_latent_parameters(self):
        "Implement here"

    @property
    def _list_of_parameters_needing_gradient(self):
        "Implement here"
    @property
    def _description(self):
        "Implement here"

    @property
    def number_of_parameters(self):
        "Implement here"
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
the `_add_doc` decorator to inherit the docstrings of the `_model` class.
