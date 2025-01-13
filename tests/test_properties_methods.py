# pylint: skip-file
from tests.generate_models import get_dict_models_fitted, get_model, get_fitted_model


from pyPLNmodels import Pln


def test_properties():
    pln = get_fitted_model("Pln", 2, "explicit")
    print("methods:", pln._useful_methods_list)
