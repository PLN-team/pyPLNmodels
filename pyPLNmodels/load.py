import os
from typing import Optional, Dict, Any, List

import pandas as pd


def load_model(path_of_directory: str) -> Dict[str, Any]:
    """
    Load Pln or PlnPCA model (that has previously been saved) from the given directory for future initialization.

    Parameters
    ----------
    path_of_directory : str
        The path to the directory containing the model.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the loaded model.
    Examples
    --------
    >>> from pyPLNmodels import PlnPCA, Pln, load_scrna, load_model
    >>> data = load_scrna()
    >>> pca = PlnPCA.from_formula("endog ~ 1", data)
    >>> pca.fit()
    >>> pca.save()
    >>> dict_init = load_model("PlnPCA_nbcov_1_dim_200_rank_5")
    >>> loaded_pca = PlnPCA.from_formula("endog ~ 1",data, dict_initialization = dict_init)
    >>> print(loaded_pca)

    >>> pln = Pln(data["endog"], add_const = True)
    >>> pln.fit()
    >>> pln.save()
    >>> dict_init = load_model("Pln_nbcov_1_dim_200")
    >>> loaded_pln = Pln(data["endog"], add_const = True, dict_initialization = dict_init)
    >>> print(loaded_pln)

    >>> from pyPLNmodels import ZIPln, load_microcosm, load_model
    >>> data = load_microcosm()
    >>> zi = ZIPln.from_formula("endog ~ 1 + time*site | 1 + time", data)
    >>> zi.fit()
    >>> zi.save("zi_model")
    >>> dict_init = load_model("zi_model")
    >>> loaded_zi = ZIPln.from_formula("endog ~ 1 + time*site | 1 + time", data, dict_initialization = dict_init)
    >>> print(loaded_zi)
    See also
    --------
    :func:`~pyPLNmodels.load_plnpcacollection`
    """
    working_dir = os.getcwd()
    try:
        os.chdir(path_of_directory)
    except FileNotFoundError as err:
        raise err(
            "The model has not been saved. Please be sure you have the right name of model."
        )
    all_files = os.listdir()
    data = {}
    for filename in all_files:
        if filename.endswith(".csv"):
            parameter = filename[:-4]
            try:
                data[parameter] = pd.read_csv(filename, header=None).values
            except pd.errors.EmptyDataError:
                print(
                    f"Can't load {parameter} since empty. Standard initialization will be performed for this parameter"
                )
    os.chdir(working_dir)
    return data


def load_pln(path_of_directory: str) -> Dict[str, Any]:
    """
    Alias for :func:`~pyPLNmodels.load.load_model`.
    """
    return load_model(path_of_directory)


def load_plnpca(path_of_directory: str) -> Dict[str, Any]:
    """
    Alias for :func:`~pyPLNmodels.load.load_model`.
    """
    return load_model(path_of_directory)


def load_plnpcacollection(
    path_of_directory: str, ranks: Optional[List[int]] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Load PlnPCAcollection models from the given directory.

    Parameters
    ----------
    path_of_directory : str
        The path to the directory containing the PlnPCAcollection models.
    ranks : List[int], optional
        A List of ranks specifying which models to load. If None, all models in the directory will be loaded.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        A dictionary containing the loaded PlnPCAcollection models, with ranks as keys.

    Raises
    ------
    ValueError
        If an invalid model name is encountered and the rank cannot be determined.

    Examples
    --------
    >>> from pyPLNmodels import PlnPCAcollection, load_scrna, load_plnpcacollection
    >>> data = load_scrna()
    >>> pcas = PlnPCAcollection.from_formula("endog ~ 1", data, ranks = [4,5,6])
    >>> pcas.fit()
    >>> pcas.save()
    >>> dict_init = load_plnpcacollection("PlnPCAcollection_nbcov_1_dim_200")
    >>> loaded_pcas = PlnPCAcollection.from_formula("endog ~ 1", data, ranks = [4,5,6], dict_of_dict_initialization = dict_init)
    >>> print(loaded_pcas)

    See also
    --------
    :func:`~pyPLNmodels.load_model`
    """
    working_dir = os.getcwd()
    os.chdir(path_of_directory)
    if ranks is None:
        dirnames = os.listdir()
        ranks = []
        for dirname in dirnames:
            try:
                rank = int(dirname[-1])
            except ValueError:
                raise ValueError(
                    f"Can't load the model {dirname}. End of {dirname} should be an int"
                )
            ranks.append(rank)
    datas = {}
    for rank in ranks:
        datas[rank] = load_model(f"PlnPCA_rank_{rank}")
    datas["ranks"] = ranks
    os.chdir(working_dir)
    return datas
