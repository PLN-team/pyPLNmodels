import warnings


def _threshold_samples_and_dim(max_samples, max_dim, n_samples, dim):
    if n_samples > max_samples:
        warnings.warn(
            f"\nTaking the whole {max_samples} samples of the "
            f"dataset (requested: n_samples={n_samples})\n"
        )
        n_samples = max_samples
    if dim > max_dim:
        warnings.warn(
            f"\nTaking the whole {max_samples} variables (requested: dim={dim})\n"
        )
        dim = max_dim
    return n_samples, dim
