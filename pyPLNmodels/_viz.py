import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms
import seaborn as sns


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
    sns.scatterplot(x=x, y=y, hue=colors, ax=ax, s=80)
    if covariances is not None:
        for i in range(covariances.shape[0]):
            _plot_ellipse(x[i], y[i], cov=covariances[i], ax=ax)
    if to_show is True:
        plt.show()
    return ax


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
