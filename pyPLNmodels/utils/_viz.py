# pylint: disable=too-many-lines
import warnings

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms, gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import networkx as nx

from pyPLNmodels.utils._utils import calculate_correlation, get_confusion_matrix

DEFAULT_TOL = 1e-6


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


def plot_correlation_arrows(axs, ccircle, variable_names):
    """
    Plot arrows representing the correlation circle.

    Parameters
    ----------
    axs : matplotlib.axes._axes.Axes
        Axes object for plotting.
    ccircle : list of tuples
        List of tuples containing correlations with the first and second principal components.
    variable_names : list
        List of names for the variables corresponding to columns in X.
    """
    for i, (corr1, corr2) in enumerate(ccircle):
        axs.arrow(
            0,
            0,
            corr1,  # 0 for PC1
            corr2,  # 1 for PC2
            lw=2,
            length_includes_head=True,
            head_width=0.05,
            head_length=0.05,
        )
        axs.text(corr1 / 2, corr2 / 2, variable_names[i])


def _viz_variables(
    pca_projected_variables, *, ax=None, colors=None, covariances: torch.Tensor = None
):
    """
    Visualize variables with a classic PCA.

    Parameters
    ----------
    pca_projected_variables: torch.Tensor
        The variables that need to be visualized.
    ax : Optional[matplotlib.axes.Axes], optional(keyword-only)
        The matplotlib axis to use. If `None`, an axis is created, by default `None`.
    colors : Optional[np.ndarray], optional(keyword-only)
        The colors to use for plotting, by default `None` (no colors).
    covariances : torch.Tensor
        Covariance of each latent distribution, of size (n_samples, 2, 2).
        If not `None`, will display ellipses with associated covariances,
        that serve as a proxy for the latent distribution. Default is `None`.
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
        colors = np.array(colors).astype(str)
    if colors is not None:
        nb_colors = len(np.unique(colors))
        if nb_colors > 15:
            sns.scatterplot(
                x=x, y=y, hue=colors, ax=ax, s=80, palette="viridis", legend=False
            )
            norm = plt.Normalize(0, nb_colors)
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])  # Required for colorbar
            plt.colorbar(sm, label="Value", ax=ax)
        else:
            sns.scatterplot(x=x, y=y, hue=colors, ax=ax, s=80, palette="viridis")
    else:
        sns.scatterplot(x=x, y=y, hue=colors, ax=ax, s=80, palette="viridis")
    if covariances is not None:
        for i in range(covariances.shape[0]):
            _plot_ellipse(x[i], y[i], cov=covariances[i], ax=ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
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
        The labels to color the samples, of size `n_samples`.
    title : str, keyword-only
        Additional title on the plot.

    Returns
    -------
    None
    """
    _, ax = plt.subplots(figsize=(10, 10))
    standardized_data = StandardScaler().fit_transform(data_matrix)
    pca_transformed_data, _ = _perform_pca(standardized_data, 2)
    pca_transformed_data = _normalize_2D(pca_transformed_data)

    _viz_variables(pca_transformed_data, ax=ax, colors=colors)
    plot_correlation_circle(
        data_matrix, variable_names, indices_of_variables, title=title, ax=ax
    )
    plt.show()


def _normalize_2D(variables):
    xs, ys = variables[:, 0], variables[:, 1]
    scalex, scaley = 1.0 / (xs.max() - xs.min()), 1.0 / (ys.max() - ys.min())
    variables[:, 0] *= scalex
    variables[:, 1] *= scaley
    return variables


def _biplot_lda(
    latent_variables, variable_names, *, clusters, indices_of_variables, title, colors
):  # pylint: disable = too-many-arguments
    _, ax = plt.subplots(figsize=(10, 10))
    transformed_lda = _get_lda_projection(latent_variables, clusters)
    transformed_lda = _normalize_2D(transformed_lda)
    _viz_variables(transformed_lda, ax=ax, colors=colors)
    data_matrix = torch.cat((latent_variables, clusters.unsqueeze(1)), dim=1)
    plot_correlation_circle(
        data_matrix,
        variable_names,
        indices_of_variables,
        title=title,
        ax=ax,
        reduction="LDA",
    )
    plt.show()


def plot_correlation_circle(
    data_matrix,
    variable_names,
    indices_of_variables,
    title="",
    ax=None,
    reduction: str = "PCA",
):  # pylint:disable=too-many-arguments, too-many-positional-arguments
    """
    Plot a correlation circle for principal component analysis (PCA).

    Parameters
    ----------
    data_matrix : np.ndarray
        Input data matrix with shape (n_samples, n_features).
        If `reduction` is "lda", the last columns should be the target

    variable_names : list
        List of names for the variables corresponding to columns in data_matrix.
    indices_of_variables : list
        List of indices of the variables to be considered in the plot.
    title : str
        Additional title on the plot.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot, by default `None`.
        If None, will call `plt.show()`
    reduction :
        Whether to use the PCA reduction or LDA reduction.
        If "LDA", the last column of data_matrix should be the clusters.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
        to_show = True
    else:
        to_show = False
    if reduction == "PCA":
        standardized_data = StandardScaler().fit_transform(data_matrix)
        transformed_data, explained_variance_ratio = _perform_pca(standardized_data, 2)
    else:
        standardized_data = StandardScaler().fit_transform(data_matrix[:, :-1])
        transformed_data, explained_variance_ratio = _perform_lda(
            data_matrix[:, :-1], data_matrix[:, -1]
        )

    correlation_circle = calculate_correlation(
        data_matrix[:, indices_of_variables], transformed_data
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_correlation_arrows(ax, correlation_circle, variable_names)

    ax.add_patch(
        Circle((0, 0), 1, facecolor="none", edgecolor="k", linewidth=1, alpha=0.5)
    )

    # Set labels and title
    ax.set_xlabel(f"{reduction} 1 ({np.round(explained_variance_ratio[0] * 100, 3)}%)")
    ax.set_ylabel(f"{reduction} 2 ({np.round(explained_variance_ratio[1] * 100, 3)}%)")
    ax.set_title(f"Correlation circle on the transformed variables {title}")
    if to_show is True:
        plt.show()


class BaseModelViz:  # pylint: disable=too-many-instance-attributes
    """Class that visualizes the parameters of a model and the optimization process."""

    def __init__(self, pln):  # pylint: disable=too-many-arguments
        self._params = pln.dict_model_parameters
        self._dict_mse = pln._dict_list_mse
        self._running_times = pln._time_recorder.running_times
        self._criterion_list = pln._elbo_criterion_monitor.criterion_list
        self._name = pln._name
        self.column_names = pln.column_names_endog
        self.n_samples = pln.n_samples

    def display_relationship_matrix(self, *, ax: matplotlib.axes.Axes):
        """
        Display a heatmap of the model covariance.
        """
        relationship_matrix = self._get_relationship_matrix()
        dim = relationship_matrix.shape[0]
        is_diagonal = len(relationship_matrix.shape) == 1
        if is_diagonal is True:
            cov_to_show = relationship_matrix.unsqueeze(0)
        else:

            if dim > 400:
                cov_to_show = relationship_matrix[:400, :400]
                warnings.warn("Only displaying the first 400 variables.")
            else:
                cov_to_show = relationship_matrix
        sns.heatmap(cov_to_show, ax=ax)
        ax.set_title(self._relationship_matrix_title)
        _set_tick_labels_columns(ax, self.column_names)

    def _get_relationship_matrix(self):
        return self._params["covariance"]

    @property
    def _relationship_matrix_title(self):
        return "Covariance matrix"

    def display_coef(self, *, ax: matplotlib.axes.Axes):
        """
        Display a heatmap of the coefficients.
        """
        coef = self._params["coef"]
        if coef is not None:
            sns.heatmap(coef, ax=ax)
        else:
            ax.text(
                0.5,
                0.5,
                "No coefficients given in the model.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        ax.set_title("Regression Coefficient Matrix", fontsize=9)
        _set_tick_labels_columns(ax, self.column_names)

    def display_norm_evolution(self, *, ax: matplotlib.axes.Axes):
        """
        Display the evolution of the norm of each parameter.
        """
        absc = np.arange(0, len(self._dict_mse[list(self._dict_mse.keys())[0]]))
        absc = absc * len(self._running_times) / len(absc)
        absc = np.array(self._running_times)[absc.astype(int)]
        for key, value in self._dict_mse.items():
            ax.plot(absc, value, label=key)
        ax.set_xlabel("Seconds", fontsize=10)
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Norm of each parameter.", fontsize=8)

    def display_criterion_evolution(self, *, ax: matplotlib.axes.Axes):
        """
        Display the criterion of the model.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on. If not provided, will be created.
        """
        ax.plot(
            self._running_times, self._criterion_list, label="Convergence Criterion"
        )

        ax.axhline(
            y=DEFAULT_TOL, color="r", linestyle="--", label="Tolerance threshold"
        )
        ax.set_yscale("log")
        ax.set_xlabel("Seconds", fontsize=9)
        ax.set_ylabel("Criterion", fontsize=9)
        ax.legend()

    def show(self, *, savefig, name_file):
        """
        Display the model parameters and the norm of the parameters.
        """
        fig = _get_figure()
        gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :])

        self.display_relationship_matrix(ax=ax1)
        self.display_norm_evolution(ax=ax2)
        self.display_criterion_evolution(ax=ax3)
        self.display_coef(ax=ax4)

        if savefig is True:
            plt.savefig(name_file + self._name + ".pdf", format="pdf")

        plt.show()


class DiagModelViz(BaseModelViz):
    """
    Model visualization class for a Pln model with diagonal covariance.
    """

    @property
    def _relationship_matrix_title(self):
        return "Variance coefficients (Covariance is diagonal)."


class LDAModelViz(BaseModelViz):
    """
    Model visualization class for a PlnLDA model.
    """

    def __init__(self, pln):
        super().__init__(pln)
        self._latent_params = pln.dict_latent_parameters
        self._exog = pln.exog
        self._clusters = pln.clusters

    @property
    def latent_positions_clusters(self):
        """
        Latent positions needed for LDA analysis.
        """
        if self._exog is None:
            return self._latent_params["latent_mean"]
        return self._latent_params["latent_mean"] - self._exog @ self._params["coef"]

    def display_norm_evolution(self, *, ax: matplotlib.axes.Axes):
        """
        Display the evolution of the norm of each parameter.
        """
        absc = np.arange(0, len(self._dict_mse[list(self._dict_mse.keys())[0]]))
        absc = absc * len(self._running_times) / len(absc)
        absc = np.array(self._running_times)[absc.astype(int)]
        for key, value in self._dict_mse.items():
            ax.plot(absc, value, label=key)
        ax.set_xlabel("Seconds", fontsize=10)
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Norm of each parameter.", fontsize=8)

    def show(self, *, savefig, name_file):
        """
        Display the model parameters and the norm of the parameters.
        """
        fig = _get_figure()
        gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1:3, :2])
        ax5 = fig.add_subplot(gs[2, 2])

        self.display_relationship_matrix(ax=ax1)
        self.display_norm_evolution(ax=ax2)
        self.display_criterion_evolution(ax=ax3)
        self.display_boundary(ax=ax4)
        self.display_coef(ax=ax5)

        if savefig is True:
            plt.savefig(name_file + self._name + ".pdf", format="pdf")
        plt.show()

    def display_boundary(self, ax):
        """Display the boundary and training points in 2D or 1D."""
        _viz_lda_train(self.latent_positions_clusters, self._clusters, ax=ax)


class ARModelViz(BaseModelViz):
    """
    Model visualization class for a Pln model with autoregressive.
    """

    def show(self, *, savefig, name_file):
        """
        Display the model parameters and the norm of the parameters.
        """
        fig = _get_figure()
        gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1:])

        self.display_relationship_matrix(ax=ax1)
        self.display_norm_evolution(ax=ax2)
        self.display_criterion_evolution(ax=ax3)
        self.display_autoreg(ax=ax4)
        self.display_coef(ax=ax5)

        if savefig is True:
            plt.savefig(name_file + self._name + ".pdf", format="pdf")

        plt.show()

    def display_autoreg(self, *, ax: matplotlib.axes.Axes):
        """
        Display a heatmap of the model covariance.
        """
        autoreg = self._params["ar_coef"]
        if autoreg.numel() == 1:
            _display_scalar_autoreg(autoreg, ax)
        else:
            if autoreg.dim() == 1:
                _display_vector_autoreg(autoreg, ax, self.column_names)
            else:
                _display_matrix_autoreg(autoreg, ax, self.column_names)

    @property
    def _relationship_matrix_title(self):
        if len(self._params["covariance"].shape) == 1:
            return "Variance coefficients (Covariance is diagonal)."
        return "Covariance Matrix"


class NetworkModelViz(BaseModelViz):
    """
    Visualize the parameters of a PlnNetwork model and the optimization process.
    """

    def _get_relationship_matrix(self):
        return torch.inverse(self._params["covariance"])

    @property
    def _relationship_matrix_title(self):
        return "Precision matrix"


class ZIModelViz(BaseModelViz):
    """
    Visualize the parameters of a ZIPln model and the optimization process.
    """

    def show(self, *, savefig, name_file):
        """
        Show the model but adds a zero inflation graph for the associated
        coefficient. Graphs are reordered so that `coef` and `coef_inflation`
        can be directly compared (compared to `show`).
        """
        fig = _get_figure()
        gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1:3])

        self.display_coef_inflation(ax=ax4)
        self.display_relationship_matrix(ax=ax2)
        self.display_norm_evolution(ax=ax5)
        self.display_criterion_evolution(ax=ax3)
        self.display_coef(ax=ax1)
        if savefig is True:
            plt.savefig(name_file + self._name + ".pdf", format="pdf")
        plt.show()

    def display_coef_inflation(self, *, ax: matplotlib.axes.Axes):
        """
        Display a heatmap of the coefficients of the zero inflation.
        """
        coef_inflation = self._params["coef_inflation"]
        sns.heatmap(coef_inflation, ax=ax)
        ax.set_title("Zero inflation Regression Coefficient Matrix")
        _set_tick_labels_columns(ax, self.column_names)


class MixtureModelViz(BaseModelViz):
    """
    Visualize the parameters of a MixturePln model and the optimization process.
    """

    def show(self, *, savefig, name_file):
        weights = self._params["weights"]
        cluster_bias = self._params["cluster_bias"]
        variances = self._params["covariances"]

        n_clusters = len(weights)

        fig = _get_figure()
        gs = gridspec.GridSpec(3, n_clusters + 2, figure=fig, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0:n_clusters])
        axes_means = [fig.add_subplot(gs[1, i]) for i in range(n_clusters)]
        axes_variances = [fig.add_subplot(gs[2, i]) for i in range(n_clusters)]
        ax3 = fig.add_subplot(gs[0, n_clusters : n_clusters + 2])
        ax4 = fig.add_subplot(gs[1, n_clusters : n_clusters + 2])
        ax5 = fig.add_subplot(gs[2, n_clusters : n_clusters + 2])

        self._plot_weights(ax1, weights)
        self._plot_cluster_biases(axes_means, cluster_bias, n_clusters)
        self._plot_variances(axes_variances, variances, n_clusters)

        self.display_norm_evolution(ax=ax3)
        self.display_criterion_evolution(ax=ax4)
        self.display_coef(ax=ax5)

        plt.tight_layout()
        plt.show()

    def _plot_weights(self, ax, weights):
        clusters = np.arange(len(weights))
        bars = ax.bar(clusters, weights, edgecolor="black")
        nb_samples = weights * self.n_samples
        ax.set_title("Histogram of Weights")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Weight")
        ax.set_xticks(clusters)
        for _bar, n_samples in zip(bars, nb_samples):
            height = _bar.get_height()
            ax.text(
                _bar.get_x() + _bar.get_width() / 2.0,
                height,
                f"{int(n_samples)}",
                ha="center",
                va="bottom",
            )

    def _plot_cluster_biases(self, axes, cluster_bias, n_clusters):
        y_indices = self.column_names
        x_min, x_max = torch.min(cluster_bias), torch.max(cluster_bias)
        for k in range(n_clusters):
            axes[k].barh(y_indices, cluster_bias[k], label=f"Cluster {k}", color="blue")
            axes[k].set_xlim(x_min, x_max)
            axes[k].set_xlabel(f"Mean of Cluster {k}", fontsize=10)
            if k > 0:
                axes[k].set_yticklabels([])

    def _plot_variances(self, axes, variances, n_clusters):
        y_indices = self.column_names
        x_max = torch.max(variances)
        for k in range(n_clusters):
            axes[k].barh(y_indices, variances[k], label=f"Cluster {k}", color="blue")
            axes[k].set_xlim(0, x_max)
            axes[k].set_xlabel(f"Variance of Cluster {k}", fontsize=10)
            if k > 0:
                axes[k].set_yticklabels([])


def _perform_pca(array, n_components):
    pca = PCA(n_components=n_components)
    proj_variables = pca.fit_transform(array)
    explained_variance = pca.explained_variance_ratio_
    return proj_variables, explained_variance


def _perform_lda(array, clusters):
    array = StandardScaler().fit_transform(array)
    lda = LinearDiscriminantAnalysis()
    proj_variables = lda.fit_transform(array, clusters)
    explained_variance = lda.explained_variance_ratio_
    return proj_variables, explained_variance


def _create_labels(explained_variance, n_components):
    return {
        str(i): f"PC{i+1}: {np.round(explained_variance * 100, 1)[i]:.1f}%"
        for i in range(n_components)
    }


def _create_dataframe(proj_variables, labels):
    data = pd.DataFrame(proj_variables)
    data.columns = labels.values()
    return data


def _plot_pairplot(data, colors):
    if colors is not None:
        nb_colors = len(np.unique(colors))
        if nb_colors > 15:
            norm = plt.Normalize(vmin=0, vmax=nb_colors)
            num_features = len(data.columns)
            fig, ax = plt.subplots(num_features, num_features, figsize=(10, 10))
            for i, icol in enumerate(data.columns):
                for j, jcol in enumerate(data.columns):
                    if i == j:  # diagonal
                        sns.distplot(data[icol], kde=False, ax=ax[i][j])
                    else:  # off diagonal
                        sm = sns.scatterplot(
                            x=data[icol],
                            y=data[jcol],
                            ax=ax[j][i],
                            hue=colors,
                            palette="viridis",
                            legend=False,
                        )
                    if i == 0:
                        ax[j][i].set_ylabel(jcol)
                    else:
                        ax[j][i].set_ylabel("")
                    if j == 2:
                        ax[j][i].set_xlabel(icol)
                    else:
                        ax[j][i].set_xlabel("")
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])  # Required for colorbar
            fig.colorbar(sm, ax=ax, orientation="vertical", label="Value")
        else:
            colors = np.array(colors)
            data["labels"] = pd.Categorical(colors, categories=pd.unique(colors))
            sns.pairplot(data, hue="labels", palette="viridis")
    else:
        sns.pairplot(data, diag_kind="kde")
    plt.show()


def _pca_pairplot(array, n_components, colors):
    """
    Generates a scatter matrix plot based on Principal Component Analysis (PCA) on
    the given array.

    Parameters
    ----------
    array: (np.ndarray): The array on which we will perform PCA and then visualize.

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
    data = _create_dataframe(proj_variables, labels)
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


def _set_axis_attributes(ax, reconstruction_error, max_y):
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
    Plot the predicted value of the `endog` against the `endog`.

    Parameters
    ----------
    endog : torch.Tensor
        The true values.
    endog_predictions : torch.Tensor
        The predicted values.
    reconstruction_error : float
        The reconstruction error.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axis to use. If `None`, the current axis is used.
    colors : np.ndarray, optional
            The labels to color the samples, of size `n_samples`.

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

    max_y = int(torch.max(torch.nan_to_num(endog_ravel)).item())
    _plot_identity_line(ax, max_y)
    _set_axis_attributes(ax, reconstruction_error, max_y)

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


def _viz_network(precision, node_labels=None, ax=None, seed=0):
    if ax is None:
        ax = plt.gca()
        to_show = True
    else:
        to_show = False
    graph, _ = _build_graph(precision, node_labels)
    pos = nx.spring_layout(graph, seed=seed)
    edges = graph.edges(data=True)
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(graph, pos, width=[d["weight"] * 2 for (u, v, d) in edges])
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=nx.get_node_attributes(graph, "label"),
        font_size=12,
        font_color="black",
    )
    plt.title("Graph Representation of Precision Matrix links")
    if to_show:
        plt.show()


def _build_graph(precision, node_labels=None):
    graph = nx.Graph()
    nb_variables = precision.shape[0]
    graph.add_nodes_from(range(nb_variables))
    if node_labels is not None:
        for i, label in enumerate(node_labels):
            graph.nodes[i]["label"] = label
    for i in range(nb_variables):
        for j in range(i + 1, nb_variables):
            if precision[i, j] != 0:
                graph.add_edge(i, j, weight=precision[i, j])

    if node_labels is not None:
        connections = {
            node_labels[node]: [
                node_labels[neighbor] for neighbor in graph.neighbors(node)
            ]
            for node in graph.nodes
        }
    else:
        connections = {node: list(graph.neighbors(node)) for node in graph.nodes}

    return graph, connections


def _viz_dims(*, variables, indices_of_variables, variable_names, colors, display):
    _, axes = plt.subplots(len(variable_names))
    absc = np.arange(variables.shape[0])
    min_y = torch.min(torch.nan_to_num(variables))
    max_y = torch.max(torch.nan_to_num(variables))
    for i, (dim, name) in enumerate(zip(indices_of_variables, variable_names)):
        y = variables[:, dim]
        sns.scatterplot(x=absc, y=y, ax=axes[i], hue=colors)
        axes[i].set_title(name)
        if display == "keep":
            axes[i].set_xlim(absc[0], absc[-1])
        axes[i].set_ylim(min_y, max_y)
    plt.show()


def _set_tick_labels_columns(ax, column_names):
    tick_positions = ax.get_xticks()
    tick_labels = [
        column_names[int(pos)] for pos in tick_positions if int(pos) < len(column_names)
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)


def _get_figure():
    return plt.figure(figsize=(20, 12))


def plot_confusion_matrix(
    pred_clusters: ArrayLike,
    true_clusters: ArrayLike,
    ax: matplotlib.axes.Axes = None,
    title: str = "",
):
    """
    Compute and plot the confusion matrix for clustering results.

    Parameters
    ----------
    pred_clusters : array-like of shape (n_samples,)
        Predicted cluster labels from k-means clustering.
    true_clusters : array-like of shape (n_samples,)
        True cluster labels.
    ax : matplotlib.axes.Axes (Optional)
        Axes object to draw the heatmap on. Default is None
    title : str (Optional)
        Title for the heatmap.
    """
    confusion_mat, pred_encoder, true_encoder = get_confusion_matrix(
        pred_clusters, true_clusters
    )
    if ax is None:
        to_show = True
        ax = plt.gca()
    else:
        to_show = False
    # sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", ax=ax)

    if pred_encoder is not None:
        pred_labels = pred_encoder.classes_
        # ax.set_xticklabels(pred_labels, rotation=45, ha="right")
    else:
        pred_labels = np.arange(confusion_mat.shape[0])

    if true_encoder is not None:
        true_labels = true_encoder.classes_
        # ax.set_yticklabels(true_labels, rotation=0)
    else:
        true_labels = np.arange(confusion_mat.shape[0])
    _show_mat(confusion_mat, pred_labels, true_labels, ax)

    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(title)

    if to_show is True:
        plt.show()


def _show_mat(mat, xlabels, ylabels, ax):
    _ = ax.imshow(mat)
    ax.set_xticks(
        range(len(xlabels)),
        labels=xlabels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(ylabels)), labels=ylabels)
    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            _ = ax.text(j, i, mat[i, j], ha="center", va="center", color="w")


def _plot_lda_2d_projection(
    *, transformed_lda_train, y_train, transformed_lda_test, colors, ax
):
    sns.scatterplot(
        x=transformed_lda_test[:, 0],
        y=transformed_lda_test[:, 1],
        hue=colors,
        palette="viridis",
        edgecolor="black",
        ax=ax,
    )
    _plot_contour_lda(transformed_lda_train, y_train, ax)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_title("LDA Projection with Decision Boundaries")


def _plot_lda_1d_projection(
    *, transformed_lda_train, y_train, transformed_lda_test, colors, ax
):
    mean_0 = transformed_lda_train[y_train == 0].mean()
    mean_1 = transformed_lda_train[y_train == 1].mean()
    boundary = (mean_0 + mean_1) / 2
    sns.kdeplot(
        transformed_lda_train[y_train == 0].ravel(),
        fill=True,
        label="Class 0 (train data)",
        alpha=0.5,
        ax=ax,
        palette="viridis",
    )
    sns.kdeplot(
        transformed_lda_train[y_train == 1].ravel(),
        fill=True,
        label="Class 1 (train data)",
        alpha=0.5,
        ax=ax,
        palette="viridis",
    )
    ax.axvline(boundary, color="black", linestyle="--", label="Decision Boundary")

    jitter = np.random.normal(0, 0.05, size=transformed_lda_test.shape)
    sns.scatterplot(
        x=transformed_lda_test.squeeze(),
        y=jitter.squeeze(),
        hue=colors,
        palette="viridis",
        edgecolor="k",
        s=80,
        alpha=0.7,
        ax=ax,
    )

    ax.legend()
    ax.set_xlabel("LDA Projection")
    ax.set_ylabel("Density (random noise on the y-axis for a better visualization)")
    ax.set_title("1D LDA Projection with Decision Boundary")


def _plot_contour_lda(transformed_lda_train, y_train, ax):
    transformed_lda_train_2d = transformed_lda_train[:, :2]
    x_min, x_max = (
        transformed_lda_train_2d[:, 0].min() - 1,
        transformed_lda_train_2d[:, 0].max() + 1,
    )
    y_min, y_max = (
        transformed_lda_train_2d[:, 1].min() - 1,
        transformed_lda_train_2d[:, 1].max() + 1,
    )
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    lda_2d = LinearDiscriminantAnalysis()
    y_train = LabelEncoder().fit_transform(y_train)
    lda_2d.fit(transformed_lda_train_2d, y_train)
    prediction = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    prediction = prediction.reshape(xx.shape)
    cmap = ListedColormap(sns.color_palette("viridis", 3).as_hex())
    ax.contourf(xx, yy, prediction, alpha=0.3, cmap=cmap)


def _get_lda_projection(X, y):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    return clf.transform(X)


def _viz_lda_train(transformed_train, y_train, ax=None):
    transformed_train = _get_lda_projection(transformed_train, y_train)
    _viz_lda(
        transformed_train=transformed_train,
        y_train=y_train,
        X_transformed=transformed_train,
        colors=y_train,
        ax=ax,
    )


def _viz_lda(*, transformed_train, y_train, X_transformed, colors, ax):
    if ax is None:
        to_show = True
        ax = plt.gca()
    else:
        to_show = False

    transformed_lda_train = _get_lda_projection(transformed_train, y_train)
    if colors is not None:
        if isinstance(colors, np.ndarray):
            if len(colors.shape) > 1:
                colors = colors.argmax(axis=1)
        if isinstance(colors, torch.Tensor):
            if len(colors.shape) > 1:
                colors = colors.argmax(dim=1)
    if len(np.unique(y_train)) > 2:
        _plot_lda_2d_projection(
            transformed_lda_train=transformed_lda_train,
            y_train=y_train,
            transformed_lda_test=X_transformed,
            colors=colors,
            ax=ax,
        )
    else:
        _plot_lda_1d_projection(
            transformed_lda_train=transformed_lda_train,
            y_train=y_train,
            transformed_lda_test=X_transformed,
            colors=colors,
            ax=ax,
        )
    if to_show:
        plt.show()


def _viz_lda_test(*, transformed_train, y_train, new_X_transformed, colors, ax=None):
    _viz_lda(
        transformed_train=transformed_train,
        y_train=y_train,
        X_transformed=new_X_transformed,
        colors=colors,
        ax=ax,
    )


def _display_matrix_autoreg(autoreg, ax, column_names):
    sns.heatmap(autoreg, ax=ax)
    ax.set_title("Autoregression coefficients")
    _set_tick_labels_columns(ax, column_names)


def _display_vector_autoreg(autoreg, ax, column_names):
    _display_matrix_autoreg(autoreg.unsqueeze(0), ax, column_names)


def _display_scalar_autoreg(autoreg, ax):
    # Ensure autoreg is a scalar between 0 and 1
    if not 0 <= autoreg <= 1:
        raise ValueError("autoreg must be a scalar between 0 and 1")

    # Plot a horizontal line from 0 to 1
    ax.plot([0, 1], [0, 0], "k-", lw=2)

    # Plot the scalar value as a point on the line
    ax.plot(autoreg, 0, "ro", label="Autoregressive coefficient")

    # Set the x-axis limits to [0, 1]
    ax.set_xlim(-0.1, 1.1)
    ax.text(0, 0, "|", ha="center", va="bottom")
    ax.text(1, 0, "|", ha="center", va="bottom")
    ax.text(0, -0.01, "Not autoregressive", ha="center", va="top")
    ax.text(1, -0.01, "Completely autoregressive", ha="center", va="top")
    ax.set_ylim(-0.1, 0.1)
    ax.set_yticks([])
    ax.legend()

    # Optionally, add labels and a title
    ax.set_title("Autoregressive coefficient")
