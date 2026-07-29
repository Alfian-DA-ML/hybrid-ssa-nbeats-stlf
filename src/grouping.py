def auto_group_deterministic(w_corr_matrix, threshold=0.9):
    """
    Automatically select deterministic reconstructed components (RCs)
    based on the weighted correlation matrix.

    RC1 (index 0) is always retained as it represents the dominant
    trend structure. Subsequent RCs are retained in adjacent pairs
    when their weighted correlation meets or exceeds the threshold,
    following the paired eigentriple structure that characterizes
    harmonic oscillations in SSA.

    Parameters
    ----------
    w_corr_matrix : np.ndarray
        Weighted correlation matrix from compute_w_correlation().
    threshold : float, default 0.9
        Minimum weighted correlation for a pair of adjacent RCs to be
        considered harmonic and retained as deterministic signal.

    Returns
    -------
    list[int]
        Sorted, deduplicated indices of the selected deterministic RCs.
    """
    idx_clean = [0]  # RC1 is always retained as the trend component

    for i in range(1, len(w_corr_matrix) - 1):
        if w_corr_matrix[i, i + 1] >= threshold:
            if i not in idx_clean:
                idx_clean.append(i)
            if (i + 1) not in idx_clean:
                idx_clean.append(i + 1)

    return sorted(set(idx_clean))


def split_trend_seasonal(idx_clean):
    """
    Split the deterministic component indices into trend and seasonal
    groups for the multichannel scenario.

    RC1 (index 0) is deterministically assigned to the trend group,
    since it consistently corresponds to the largest eigenvalue.
    All other selected components are assigned to the seasonal group.

    Parameters
    ----------
    idx_clean : list[int]
        Indices returned by auto_group_deterministic().

    Returns
    -------
    tuple[list[int], list[int]]
        (idx_trend, idx_seasonal)
    """
    idx_trend = [0]
    idx_seasonal = [idx for idx in idx_clean if idx != 0]
    return idx_trend, idx_seasonal