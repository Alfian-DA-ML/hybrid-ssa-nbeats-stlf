import numpy as np


def compute_w_correlation(components, window_length):
    """
    Compute the weighted correlation (w-correlation) matrix between
    SSA reconstructed components (RCs).

    Parameters
    ----------
    components : np.ndarray, shape (n_components, N)
        Reconstructed components from SSA decomposition.
    window_length : int
        SSA window length (L) used during decomposition.

    Returns
    -------
    np.ndarray, shape (n_components, n_components)
        Weighted correlation matrix.
    """
    L = window_length
    N = components.shape[1]
    w = np.array([min(k, L, N - k + 1) for k in range(1, N + 1)])
    weighted_comps = components * w
    inner_product = np.dot(weighted_comps, components.T)
    norms = np.sqrt(np.diag(inner_product))
    norm_matrix = np.outer(norms, norms)
    w_corr = np.abs(inner_product / norm_matrix)
    return w_corr