import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms, gridspec
from matplotlib.patches import Circle
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from pyPLNmodels._utils import calculate_correlation


def _plot_ellipse(mean_x: float, mean_y: float, *, cov: np.ndarray, ax) -> float:
    """
    Plot an ellipse given two coordinates and
    the covariance (as a 2 x 2 positive definite matrix).

    Parameters:
    -----------
    mean_x : float
        x-coordinate of the mean.
    mean_y : float
        y-coordinate of the mean.
    cov : np.ndarray (keyword-only)
        Covariance matrix of the 2d vector.
    ax : object (keyword-only)
        Axes object to plot the ellipse on.
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        linestyle="--",
        alpha=0.2,
    )

    scale_x = np.sqrt(cov[0, 0])
    scale_y = np.sqrt(cov[1, 1])
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def plot_correlation_arrows(axs, ccircle, variables_names):
    """
    Plot arrows representing the correlation circle.

    Parameters
    ----------
    axs : matplotlib.axes._axes.Axes
        Axes object for plotting.
    ccircle : list of tuples
        List of tuples containing correlations with the first and second principal components.
    variables_names : list
        List of names for the variables corresponding to columns in X.
    """
    for i, (corr1, corr2) in enumerate(ccircle):
        axs.arrow(
            0,
            0,
            corr1,  # 0 for PC1
            corr2,  # 1 for PC2
            lw=2,  # line width
            length_includes_head=True,
            head_width=0.05,
            head_length=0.05,
        )
        axs.text(corr1 / 2, corr2 / 2, variables_names[i])


def _viz_variables(
    pca_projected_variables, *, ax=None, colors=None, covariances: torch.Tensor = None
):
    """
    Visualize variables with a classic PCA.

    Parameters
    ----------
    pca_projected_variables: torch.Tensor
        The variables that need to be visualize
    ax : Optional[matplotlib.axes.Axes], optional(keyword-only)
        The matplotlib axis to use. If None, an axis is created, by default None.
    colors : Optional[np.ndarray], optional(keyword-only)
        The colors to use for plotting, by default None (no colors).
    covariances : torch.Tensor
        Covariance of each latent distribution, of size (n_samples, 2, 2)
        If not None, will display ellipses with associated covariances,
        that serves as a proxy for the latent distribution. Default is None.
    Raises
    ------
    Returns
    -------
    Any
        The matplotlib axis.
    """
    x = pca_projected_variables[:, 0]
    y = pca_projected_variables[:, 1]
    if ax is None:
        ax = plt.gca()
        to_show = True
    else:
        to_show = False
    if colors is not None:
        if isinstance(colors, np.ndarray):
            colors = np.squeeze(colors)

    sns.scatterplot(x=x, y=y, hue=colors, ax=ax, s=80)
    if covariances is not None:
        for i in range(covariances.shape[0]):
            _plot_ellipse(x[i], y[i], cov=covariances[i], ax=ax)
    if to_show is True:
        plt.show()
    return ax


def _biplot(data_matrix, variable_names, *, indices_of_variables, colors, title):
    """
    Create a biplot that combines a scatter plot of PCA projected variables
    and a correlation circle.

    Parameters
    ----------
    data_matrix : np.ndarray
        Input data matrix with shape (n_samples, n_features).
    variable_names : list
        List of names for the variables corresponding to columns in data_matrix.
    indices_of_variables : list, keyword-only
        List of indices of the variables to be considered in the plot.
    colors : Optional[np.ndarray], optional, keyword-only
        The colors to use for plotting, by default None (no colors).
    title : str, keyword-only
        Additional title on the plot.

    Returns
    -------
    None
    """
    standardized_data = StandardScaler().fit_transform(data_matrix)
    pca_transformed_data, explained_variance_ratio = _perform_pca(standardized_data, 2)

    pca_projected_variables = torch.tensor(pca_transformed_data)
    xs, ys = pca_projected_variables[:, 0], pca_projected_variables[:, 1]
    scalex, scaley = 1.0 / (xs.max() - xs.min()), 1.0 / (ys.max() - ys.min())
    pca_projected_variables[:, 0] *= scalex
    pca_projected_variables[:, 1] *= scaley

    _, ax = plt.subplots(figsize=(10, 10))
    _viz_variables(pca_projected_variables, ax=ax, colors=colors)

    correlation_circle = calculate_correlation(
        data_matrix[:, indices_of_variables], pca_transformed_data
    )

    plot_correlation_arrows(ax, correlation_circle, variable_names)

    ax.add_patch(
        Circle((0, 0), 1, facecolor="none", edgecolor="k", linewidth=1, alpha=0.5)
    )

    ax.set_xlabel(f"PCA 1 ({np.round(explained_variance_ratio[0] * 100, 3)}%)")
    ax.set_ylabel(f"PCA 2 ({np.round(explained_variance_ratio[1] * 100, 3)}%)")
    ax.set_title(f"Biplot: {title}")

    plt.show()


def plot_correlation_circle(
    data_matrix, variable_names, indices_of_variables, title=""
):
    """
    Plot a correlation circle for principal component analysis (PCA).

    Parameters
    ----------
    data_matrix : np.ndarray
        Input data matrix with shape (n_samples, n_features).
    variable_names : list
        List of names for the variables corresponding to columns in data_matrix.
    indices_of_variables : list
        List of indices of the variables to be considered in the plot.
    title : str
        Additional title on the plot.
    """
    standardized_data = StandardScaler().fit_transform(data_matrix)
    pca_transformed_data, explained_variance_ratio = _perform_pca(standardized_data, 2)

    correlation_circle = calculate_correlation(
        data_matrix[:, indices_of_variables], pca_transformed_data
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(10, 10))
    plot_correlation_arrows(ax, correlation_circle, variable_names)

    unit_circle = Circle(
        (0, 0), 1, facecolor="none", edgecolor="k", linewidth=1, alpha=0.5
    )
    ax.add_patch(unit_circle)

    # Set labels and title
    ax.set_xlabel(f"PCA 1 ({np.round(explained_variance_ratio[0] * 100, 3)}%)")
    ax.set_ylabel(f"PCA 2 ({np.round(explained_variance_ratio[1] * 100, 3)}%)")
    ax.set_title(f"Correlation circle on the transformed variables {title}")

    plt.show()


class ModelViz:
    """Class that visualize the parameters of a model and the optimization process."""

    def __init__(
        self, *, params, dict_mse, running_times, criterion_list, name, tol
    ):  # pylint: disable=too-many-arguments
        self._params = params
        self._dict_mse = dict_mse
        self._running_times = running_times
        self._criterion_list = criterion_list
        self._name = name
        self._tol = tol

    def display_covariance(self, *, ax: matplotlib.axes.Axes):
        """
        Display an heatmap of the model covariance.
        """
        covariance = self._params["covariance"]
        dim = covariance.shape[0]
        if dim > 400:
            cov_to_show = covariance[:400, :400]
            warnings.warn("Only displaying the first 400 variables.")
        else:
            cov_to_show = covariance
        sns.heatmap(cov_to_show, ax=ax)
        ax.set_title("Covariance matrix")

    def display_coef(self, *, ax: matplotlib.axes.Axes):
        """
        Display an heatmap of the coefficient.
        """
        coef = self._params["coef"]
        if coef is not None:
            sns.heatmap(coef, ax=ax)
        else:
            ax.text(
                0.5,
                0.5,
                "No coef given in the model.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        ax.set_title("Regression coefficient Matrix")

    def display_norm_evolution(self, *, ax: matplotlib.axes.Axes):
        """
        Display the evolution of the norm of each parameter.
        """
        absc = np.arange(0, len(self._dict_mse[list(self._dict_mse.keys())[0]]))
        absc = absc * len(self._running_times) / len(absc)
        absc = np.array(self._running_times)[absc.astype(int)]
        for key, value in self._dict_mse.items():
            ax.plot(absc, value, label=key)
        ax.set_xlabel("Seconds", fontsize=15)
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Norm of each parameter.")

    def display_criterion_evolution(self, *, ax: matplotlib.axes.Axes):
        """
        Display the criterion of the model.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on. If not provided, will be created.
        """
        ax.plot(self._running_times, self._criterion_list, label="Criterion")

        # last_criterion = np.round(self._criterion_list[-1], 6)
        last_criterion = f"{np.round(self._criterion_list[-1], 6):.2e}"
        ax.axhline(y=self._tol, color="r", linestyle="--", label="Tolerance threshold")
        ax.set_title(f"Criterion. Last criterion = {last_criterion}", fontsize=14)
        ax.set_yscale("log")
        ax.set_xlabel("Seconds", fontsize=14)
        ax.set_ylabel("Criterion", fontsize=14)
        ax.legend()

    def show(self, *, axes, savefig, name_file):
        """
        Display the model parameters and the norm of the parameters.
        """
        to_show = False
        if axes is None:
            fig = plt.figure(figsize=(23, 5))
            gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])

            ax4 = fig.add_subplot(gs[1, :])
            to_show = True
        else:
            try:
                ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]
            except IndexError as err:
                error_message = "You should be able to access the axes using axes[3]."
                print(error_message)
                raise IndexError(f"{error_message}: {err}") from err

        self.display_covariance(ax=ax1)
        self.display_norm_evolution(ax=ax2)
        self.display_criterion_evolution(ax=ax3)
        self.display_coef(ax=ax4)

        if savefig is True:
            plt.savefig(name_file + self._name + ".pdf", format="pdf")

        if to_show is True:
            plt.show()

    def show_zi(self, *, axes, savefig, name_file):
        """
        Show the model but adds a zero inflation graph for the associated
        coefficient. Graphs are reordered so that `coef` and `coef_inflation`
        can be directly compared (compared to `show`).
        """
        to_show = False
        if axes is None:
            fig = plt.figure(figsize=(23, 5))
            gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])

            ax4 = fig.add_subplot(gs[1, 0])
            ax5 = fig.add_subplot(gs[1, 1:3])
            to_show = True
            # axes = [ax4,ax1,ax3,ax2]
            axes = [ax2, ax5, ax3, ax1]
        else:
            try:
                ax1, ax2, ax3, ax4, ax5 = axes[0], axes[1], axes[2], axes[3], axes[4]
            except IndexError as err:
                error_message = "You should be able to access the axes using axes[4]."
                print(error_message)
                raise IndexError(f"{error_message}: {err}") from err

        self.show(axes=axes, savefig=False, name_file="")
        self.display_coef_inflation(ax=ax4)
        if savefig is True:
            plt.savefig(name_file + self._name + ".pdf", format="pdf")

        if to_show is True:
            plt.show()

    def display_coef_inflation(self, *, ax: matplotlib.axes.Axes):
        """
        Display an heatmap of the coefficient of the zero inflation.
        """
        coef_inflation = self._params["coef_inflation"]
        sns.heatmap(coef_inflation, ax=ax)
        ax.set_title("Zero inflation Regression coefficient Matrix")


def _perform_pca(array, n_components):
    pca = PCA(n_components=n_components)
    proj_variables = pca.fit_transform(array)
    explained_variance = pca.explained_variance_ratio_
    return proj_variables, explained_variance


def _create_labels(explained_variance, n_components):
    return {
        str(i): f"PC{i+1}: {np.round(explained_variance * 100, 1)[i]:.1f}%"
        for i in range(n_components)
    }


def _create_dataframe(proj_variables, labels, colors):
    data = pd.DataFrame(proj_variables)
    data.columns = labels.values()
    if colors is not None:
        data["labels"] = colors
    return data


def _plot_pairplot(data, colors):
    if colors is not None:
        sns.pairplot(data, hue="labels", diag_kind="bins")
    else:
        sns.pairplot(data, diag_kind="kde")
    plt.show()


def _pca_pairplot(array, n_components, colors):
    """
    Generates a scatter matrix plot based on Principal Component Analysis (PCA) on
    the given array.

    Parameters
    ----------
    array: (np.ndarray): The array on which we will perform pca and then visualize.

    n_components (int, optional): The number of components to consider for plotting.
        If not specified, the maximum number of components will be used. Note that
        it will not display more than 10 graphs.
        Defaults to None.

    colors (np.ndarray): An array with one label for each
        sample in the endog property of the object.
        Defaults to None.
    Raises
    ------
    ValueError: If the number of components requested is greater
        than the number of variables in the dataset.
    """
    proj_variables, explained_variance = _perform_pca(array, n_components)
    labels = _create_labels(explained_variance, n_components)
    data = _create_dataframe(proj_variables, labels, colors)
    _plot_pairplot(data, colors)


def _prepare_axis(ax):
    if ax is None:
        ax = plt.gca()
        to_show = True
    else:
        to_show = False
    return ax, to_show


def _prepare_colors(colors, dim):
    if colors is not None:
        colors = np.repeat(np.array(colors), repeats=dim).ravel()
    return colors


def _plot_scatter(ax, endog, predictions, colors):
    sns.scatterplot(x=endog, y=predictions, hue=colors, ax=ax)


def _plot_identity_line(ax, max_y):
    y = np.linspace(0, max_y, max_y)
    ax.plot(y, y, c="red")


def _set_axis_properties(ax, reconstruction_error, max_y):
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(
        f"Reconstruction error (RMSE): {np.round(reconstruction_error.item(), 3)}"
    )
    ax.set_ylabel("Predicted values")
    ax.set_xlabel("Counts")
    ax.set_ylim(top=max_y, bottom=0.001)
    ax.legend()


def _plot_expected_vs_true(
    endog,
    endog_predictions,
    reconstruction_error,
    ax: matplotlib.axes.Axes,
    colors: np.ndarray,
):
    """
    Plot the predicted value of the endog against the endog.

    Parameters
    ----------
    endog : torch.Tensor
        The true values.
    endog_predictions : torch.Tensor
        The predicted values.
    reconstruction_error : float
        The reconstruction error.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axis to use. If None, the current axis is used.
    colors : np.ndarray, optional
        The colors to use for plotting.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axis.
    """
    ax, to_show = _prepare_axis(ax)
    predictions = endog_predictions.ravel().detach()
    colors = _prepare_colors(colors, endog.shape[1])
    endog_ravel = endog.ravel()

    _plot_scatter(ax, endog_ravel, predictions, colors)

    max_y = int(torch.max(endog_ravel).item())
    _plot_identity_line(ax, max_y)
    _set_axis_properties(ax, reconstruction_error, max_y)

    if to_show:
        plt.show()
    return ax


def _show_information_criterion(*, bic, aic, loglikes):
    colors = {"BIC": "blue", "AIC": "red", "Negative log likelihood": "orange"}

    best_bic_rank = list(bic.keys())[np.argmin(list(bic.values()))]
    best_aic_rank = list(aic.keys())[np.argmin(list(aic.values()))]

    criteria = ["BIC", "AIC", "Negative log likelihood"]
    values_list = [bic, aic, loglikes]

    for criterion, values in zip(criteria, values_list):
        plt.scatter(
            values.keys(),
            values.values(),
            label=f"{criterion} criterion",
            c=colors[criterion],
        )
        plt.plot(values.keys(), values.values(), c=colors[criterion])

        if criterion == "BIC":
            plt.axvline(best_bic_rank, c=colors[criterion], linestyle="dotted")
        elif criterion == "AIC":
            plt.axvline(best_aic_rank, c=colors[criterion], linestyle="dotted")

        plt.xticks(list(values.keys()))

    plt.legend()
    plt.show()
