# pylint: skip-file
from pyPLNmodels import load_crossover, PlnAR


def test_viz_dims():
    data = load_crossover(n_samples=2500)
    ar = PlnAR(data["endog"], add_const=True)
    ar.fit()
    ar.viz_dims(
        variables_names=["nco_Lacaune_F", "nco_Lacaune_M"], colors=data["chrom"]
    )
