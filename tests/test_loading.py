# pylint: skip-file
from pyPLNmodels import (
    load_scrna,
    load_microcosm,
    load_oaks,
    load_crossover,
    load_crossover_per_chromosom,
)


def test_right_shape():
    rna = load_scrna()
    assert rna["endog"].shape == (400, 100)
    dim = 20
    n_samples = 50
    rna_low = load_scrna(dim=dim, n_samples=n_samples)
    assert rna_low["endog"].shape == (50, 20)
    micro = load_microcosm()
    assert micro["endog"].shape == (400, 100)
    dim = 20
    n_samples = 50
    micro_low = load_microcosm(dim=dim, n_samples=n_samples)
    assert micro_low["endog"].shape == (50, 20)

    oaks = load_oaks()
    assert oaks["endog"].shape == (116, 114)
    crossover_1_chrom = load_crossover(chromosome_numbers=1)

    per_species = load_crossover_per_chromosom()
    assert per_species["endog"].shape == (276, 104)

    per_species = load_crossover_per_chromosom(n_samples=300, dim=500)
