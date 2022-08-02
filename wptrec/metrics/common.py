import numpy as np

work_order = [
    'Stub',
    'Start',
    'C',
    'B',
    'GA',
    'FA',
]

def discount(n_or_ranks):
    """
    Compute the discount function.

    Args:
        n_or_ranks(int or array-like):
            If an integer, the number of entries to discount; if an array,
            the ranks to discount.
    Returns:
        numpy.ndarray:
            The discount for the specified ranks.
    """
    if isinstance(n_or_ranks, int):
        n_or_ranks = np.arange(1, n_or_ranks + 1, dtype='f4')
    else:
        n_or_ranks = np.require(n_or_ranks, 'f4')
    n_or_ranks = np.maximum(n_or_ranks, 2)
    disc = np.log2(n_or_ranks)
    return np.reciprocal(disc)

