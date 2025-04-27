import pkg_resources
import pandas as pd
import numpy as np


from pyPLNmodels.load_data.utils import _threshold_samples_and_dim


def load_crossover(n_samples=500, *, chromosome_numbers=range(1, 27)):
    """
    Load crossover data. It contains 2459 samples (regions) with 4 dimensions (species).
    Each sample belongs to a certain chromosome, ranging from 1 to 26. The length of each chromosome
    varies.

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
        Number of samples to load, by default 500 (maximum is 2459).
    chromosome_numbers : int or range, optional
        Chromosome numbers to filter, by default range(1, 27).

    Returns
    -------
    dict
        Dictionary containing:
        - 'endog':    DataFrame with endogenous variables.
        - 'chrom':      Series with chromosome numbers.
        - 'chrom_1hot': DataFrame with one-hot encoded chromosome numbers.
        - 'offsets':  DataFrame with coverage offsets (in log scale).
        - 'location': The location of the crossover counts

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

    location = (
        (data["wstart"] / 1000000).astype(int).astype(str)
        + "-"
        + (data["wstop"] / 1000000).astype(int).astype(str)
    )
    data["location"] = location

    detailed_location = "chrom" + data["chrom"].astype(str) + "Loc:" + location
    data["detailed_location"] = detailed_location
    data.set_index("detailed_location", inplace=True)

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

    chrom = data["chrom"].astype(str)
    chrom_1hot = pd.get_dummies(chrom)
    chrom_1hot.columns = "Chr " + chrom_1hot.columns.astype(str)

    return {
        "endog": endog,
        "chrom": chrom,
        "chrom_1hot": chrom_1hot,
        "offsets": np.log(offsets),
        "location": location,
    }


def load_crossover_per_chromosom(n_samples=276, dim=104):
    """
    Load crossover data. It contains 247 samples (regions) with 104 dimensions
    (26 chromosome * 4 species).

    This dataset describes recombination patterns in sheep, focusing on the genetic determinism
    of recombination. Recombination is a biological process during which chromosomes exchange
    genetic material, leading to genetic diversity. The dataset includes male recombination maps
    for the Lacaune breed and combines results from Lacaune and Soay sheep to create precise male
    meiotic recombination maps. The dataset identifies ∼50,000 crossover hotspots (regions where
    recombination occurs frequently) and highlights major loci (specific locations on chromosomes)
    affecting recombination rate variation.

    Some chromosomes are shorter than other, resulting in NaNs.

    References:
        Petit, Morgane, Jean-Michel Astruc, Julien Sarry, Laurence Drouilhet,
        Stéphane Fabre, Carole R Moreno, and Bertrand Servin. 2017. “Variation
        in Recombination Rate and Its Genetic Determinism in Sheep Populations.”
        Genetics 207 (2): 767–84.
        https://doi.org/10.1534/genetics.117.300123

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to load, by default 276 (maximum).
    dim: int, optional
        Number of dimensions to load, by default 104 (maximum)

    Returns
    -------
    dict
        Dictionary containing:
        - 'endog':    DataFrame with endogenous variables.
        - 'offsets':  DataFrame with coverage offsets (in log scale).

    Examples
    --------
    >>> from pyPLNmodels import load_crossover_per_chromosom
    >>> data = load_crossover_per_chromosom()
    >>> print('Keys: ', data.keys())
    >>> print(data["endog"].head())
    >>> print(data["endog"].describe())

    Notes
    -----
    The very first sample may begin with nan, and is replaced by the value of the second
    sample which is not nan.
    """
    max_samples, max_dim = 276, 104
    n_samples, dim = _threshold_samples_and_dim(max_samples, max_dim, n_samples, dim)
    data_stream = pkg_resources.resource_stream(
        __name__, "data/crossover/crossover_wide.csv"
    )
    data = pd.read_csv(data_stream).drop(columns="Unnamed: 0")
    data["wstart"] /= 1000000
    data["wstop"] /= 1000000
    data["loc"] = (
        "Loc:"
        + data["wstart"].astype(int).astype(str)
        + "-"
        + data["wstop"].astype(int).astype(str)
    )
    data = data.drop(columns=["wstop"])
    data_pivoted = data.pivot_table(index=["loc", "wstart"], columns="chrom")
    data_pivoted.columns = [f"chr_{chrom}:{val}" for val, chrom in data_pivoted.columns]
    data_pivoted = data_pivoted.sort_values(by="wstart")
    data_pivoted = data_pivoted.reset_index(level="wstart", drop=True)
    endog = data_pivoted.iloc[
        :, ["nco" in column for column in data_pivoted.columns]
    ].iloc[:n_samples, :dim]
    isnan_first_sample = np.isnan(endog.iloc[0]).values
    endog.iloc[0, isnan_first_sample] = endog.iloc[1, isnan_first_sample]
    offsets = data_pivoted.iloc[
        :, ["coverage" in column for column in data_pivoted.columns]
    ].iloc[:n_samples, :dim]
    offsets.iloc[0, isnan_first_sample] = offsets.iloc[1, isnan_first_sample]
    offsets = np.log(offsets)
    return {"endog": endog, "offsets": offsets}
