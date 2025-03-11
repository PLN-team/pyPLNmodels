# pylint: skip-file
import pytest
from pyPLNmodels import load_crossover, PlnAR


def test_viz_dims():
    data = load_crossover(n_samples=2500)
    ar = PlnAR(data["endog"], add_const=True)
    ar.fit()
    ar.viz_dims(column_names=["nco_Lacaune_F", "nco_Lacaune_M"], colors=data["chrom"])
    ar.viz_dims(
        column_names=["nco_Lacaune_F", "nco_Lacaune_M"],
        colors=data["chrom"],
        display="keep",
    )
    print(ar)
    with pytest.raises(ValueError):
        ar.viz_dims(
            column_names=["nco_Lacaune_F", "nco_Lacaune_M"],
            colors=data["chrom"],
            display="wrong display",
        )
    with pytest.raises(AttributeError):
        PlnAR(data["endog"], ar_type="wrong_ar")
