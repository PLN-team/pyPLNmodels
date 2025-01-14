# pylint: skip-file
from pyPLNmodels import Pln

from tests.generate_models import get_dict_models_fitted, get_model, get_fitted_model


def test_method_properties():
    pln = get_fitted_model("Pln", 2, "explicit")
    for method, attribute in zip(
        pln._useful_methods_list, pln._useful_properties_list
    ):  # pylint: disable=protected-access
        method = method[1:-2]
        attribute = attribute[1:]
        assert hasattr(pln, attribute)
        assert hasattr(pln, method)
