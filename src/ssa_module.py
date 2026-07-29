"""
Core Singular Spectrum Analysis (SSA) engine.

Implements embedding, decomposition, and reconstruction using the
Broomhead-King trick (eigendecomposition of X @ X.T instead of full
SVD of X) for efficiency on long time series, plus vectorized
diagonal averaging.
"""

import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view


def embed(series, window_length):
    """
    Build the trajectory matrix via a sliding window embedding.

    Uses stride_tricks (a virtual view, no copy) for memory efficiency.

    Parameters
    ----------
    series : array-like
        1D input time series.
    window_length : int
        SSA window length (L).

    Returns
    -------
    np.ndarray, shape (L, K)
        Trajectory matrix, where K = N - L + 1.
    """
    X = sliding_window_view(series, window_length).T
    return X


def decompose(X):
    """
    Decompose the trajectory matrix via eigendecomposition of the
    covariance matrix X @ X.T (Broomhead-King trick).

    This computes SVD on an (L x L) matrix rather than the full
    (L x K) trajectory matrix, which is far cheaper when K >> L
    (long time series with a modest window length).

    Parameters
    ----------
    X : np.ndarray, shape (L, K)
        Trajectory matrix from embed().

    Returns
    -------
    U : np.ndarray, shape (L, L)
        Eigenvectors, sorted by descending eigenvalue.
    s : np.ndarray, shape (L,)
        Singular values (sqrt of eigenvalues), descending.
    None
        Vt is not computed explicitly; reconstruct() projects using
        U and the original X instead, which avoids the memory cost
        of a full (L, K) Vt matrix.
    """
    L, K = X.shape
    Cov = X @ X.T

    eigvals, U = np.linalg.eigh(Cov)

    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    U = U[:, idx]

    s = np.sqrt(np.abs(eigvals))

    return U, s, None


def diagonal_averaging(X):
    """
    Reconstruct a 1D series from a (possibly rank-reduced) trajectory
    matrix via diagonal averaging (Hankelization).

    Vectorized over the K dimension: loops only over the window
    length L (small), not over the full series length (large).

    Parameters
    ----------
    X : np.ndarray, shape (L, K)

    Returns
    -------
    np.ndarray, shape (N,)
        Reconstructed 1D series, where N = L + K - 1.
    """
    L, K = X.shape
    N = L + K - 1

    result = np.zeros(N)
    counts = np.zeros(N)

    for i in range(L):
        result[i : i + K] += X[i, :]
        counts[i : i + K] += 1

    return result / counts


def reconstruct(U, s, Vt, indices, X_input=None):
    """
    Reconstruct one or more SSA components by projecting the
    trajectory matrix onto the subspace spanned by the selected
    eigenvectors, then diagonal-averaging back to a 1D series.

    Parameters
    ----------
    U : np.ndarray
        Eigenvectors from decompose().
    s : np.ndarray
        Singular values from decompose() (unused directly here, kept
        for API symmetry).
    Vt : None
        Not used; kept for API symmetry with a conventional SVD
        interface. The projection is computed directly from
        X_input instead.
    indices : list[int]
        Indices of the components to reconstruct.
    X_input : np.ndarray, shape (L, K)
        The original trajectory matrix (required for projection).

    Returns
    -------
    np.ndarray, shape (N,)
        Reconstructed 1D series for the selected component(s).
    """
    U_sub = U[:, indices]

    PC = U_sub.T @ X_input
    X_rec = U_sub @ PC

    return diagonal_averaging(X_rec)


def SSA(series, window_length, return_time=False):
    """
    Run the full SSA pipeline: embed, decompose, and reconstruct
    every individual component (RC1, RC2, ..., RC_L).

    Parameters
    ----------
    series : array-like
        1D input time series.
    window_length : int
        SSA window length (L).
    return_time : bool, default False
        If True, also return the elapsed wall-clock time in seconds.

    Returns
    -------
    components : np.ndarray, shape (L, N)
        Reconstructed components, one row per component, ordered by
        descending singular value (component 0 = dominant trend).
    s : np.ndarray, shape (L,)
        Singular values, descending.
    elapsed : float, optional
        Only returned if return_time=True.
    """
    start = time.time()

    series = np.array(series)

    X = embed(series, window_length)
    U, s, _ = decompose(X)

    components = []
    for i in range(len(s)):
        comp = reconstruct(U, s, None, [i], X_input=X)
        components.append(comp)

    elapsed = time.time() - start

    if return_time:
        return np.array(components), s, elapsed
    else:
        return np.array(components), s
