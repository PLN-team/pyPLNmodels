import pkg_resources
import pandas as pd
import numpy as np


from pyPLNmodels.load_data.utils import _threshold_samples_and_dim


def load_crossover(n_samples=300, *, chromosome_numbers=range(1, 27)):
    """
    Load crossover data.

    This dataset describes recombination patterns in sheep, focusing on the genetic determinism
    of recombination. Recombination is a biological process during which chromosomes exchange
    genetic material, leading to genetic diversity. The dataset includes male recombination maps
    for the Lacaune breed and combines results from Lacaune and Soay sheep to create precise male
    meiotic recombination maps. The dataset identifies ∼50,000 crossover hotspots (regions where
    recombination occurs frequently) and highlights major loci (specific locations on chromosomes)
    affecting recombination rate variation.

    References:
        Petit, Morgane, Jean-Michel Astruc, Julien Sarry, Laurence Drouilhet,
        Stéphane Fabre, Carole R Moreno, and Bertrand Servin. 2017. “Variation
        in Recombination Rate and Its Genetic Determinism in Sheep Populations.”
        Genetics 207 (2): 767–84.
        https://doi.org/10.1534/genetics.117.300123

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to load, by default 300.
    chromosome_numbers : int or range, optional
        Chromosome numbers to filter, by default range(1, 27).

    Returns
    -------
    dict
        Dictionary containing:
        - 'endog':    DataFrame with endogenous variables.
        - 'chrom':      Series with chromosome numbers.
        - 'chrom_1hot': DataFrame with one-hot encoded chromosome numbers.
        - 'offsets':  DataFrame with coverage offsets.

    Examples
    --------
    >>> from pyPLNmodels import load_crossover
    >>> data = load_crossover()
    >>> print('Keys: ', data.keys())
    >>> print(data["endog"].head())
    >>> print(data["endog"].describe())
    """
    max_samples = 2459
    n_samples, _ = _threshold_samples_and_dim(max_samples, 4, n_samples, 4)

    data_stream = pkg_resources.resource_stream(
        __name__, "data/crossover/crossover_wide.csv"
    )
    data = pd.read_csv(data_stream).drop(columns="Unnamed: 0")

    if isinstance(chromosome_numbers, int):
        data = data[data["chrom"] == chromosome_numbers]
    else:
        data = data[np.isin(data["chrom"], chromosome_numbers)]
    data = data.iloc[:n_samples]

    endog = data[["nco_Lacaune_M", "nco_Lacaune_F", "nco_Soay_F", "nco_Soay_M"]].copy()
    print(f"Returning crossover dataset of size {endog.shape}")
    offsets = data[
        [
            "coverage_Lacaune_M",
            "coverage_Lacaune_F",
            "coverage_Soay_F",
            "coverage_Soay_M",
        ]
    ]

    endog["Location"] = (
        "chrom"
        + data["chrom"].astype(str)
        + "Loc:"
        + (data["wstart"] / 1000000).astype(int).astype(str)
        + "-"
        + (data["wstop"] / 1000000).astype(int).astype(str)
    )
    endog.set_index("Location", inplace=True)

    chrom = data["chrom"]
    chrom_1hot = pd.get_dummies(chrom)
    chrom_1hot.columns = "Chr " + chrom_1hot.columns.astype(str)

    return {
        "endog": endog,
        "chrom": chrom,
        "chrom_1hot": chrom_1hot,
        "offsets": offsets,
    }
