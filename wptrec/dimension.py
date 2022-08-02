"""
Logic for working with fairness definitions.
"""
from math import prod
import pandas as pd
import xarray as xr
import numpy as np

class FairDim:
    name: str
    page_align_df: pd.DataFrame
    page_align_xr: xr.DataArray
    global_tgt: pd.Series
    has_unknown: bool

    def __init__(self, df, xr, tgt=None, unk=False):
        self.name = df.name
        self.page_align_df = df
        self.page_align_xr = xr
        self.global_tgt = tgt
        self.has_unknown = unk

    @property
    def py_name(self):
        return self.name.replace('-', '_')

    def __str__(self):
        return f'<dimension "{self.name}": {self.page_align_df.shape[1]} levels>'
    
    def __repr__(self):
        return str(self)


def combine_alignments(arrays, exclude=['page', 'topic-id', 'topic_id']):
    "Combine alignment xarrays in a hopefully-optimized way"
    
    # helper function
    def size(a):
        sizes = [sz for (n, sz) in zip(a.dims, a.shape) if n not in exclude]
        return prod(sizes)

    work = list(arrays)
    sizes = [size(a) for a in work]
    while len(work) > 1:
        assert len(work) == len(sizes)
        # find the smallest pair
        min_i = 0
        min_sz = None
        for i, j in zip(range(len(work)), range(1, len(work))):
            ps = sizes[i] * sizes[j]
            if min_sz is None or ps < min_sz:
                min_sz = ps
                min_i = i
        
        a1, a2 = work[min_i:min_i+2]
        ap = a1 * a2
        work[min_i:min_i+2] = [ap]
        sizes[min_i:min_i+2] = [size(ap)]

    assert len(work) == 1
    return work[0]


@njit(parallel=True)
def _sum_outer_products(A, B):
    assert A.shape[0] == B.shape[0]
    n = A.shape[0]
    
    sz_a = A.size // n
    sz_b = B.size // n

    Ap = A.reshape(n, sz_a)
    Bp = B.reshape(n, sz_b)
    
    out = np.zeros((sz_a, sz_b))
    for i in range(n):
        Ar = A[i].ravel()
        Br = B[i].ravel()
        for j in prange(Ar.size):
            out[j] += Ar[j] * Br
    
    return out.reshape(A.shape[1:] + B.shape[1:])


def _mean_outer_ufunc(A, B):
    n = A.shape[-1]
    sz_a = A.size // n
    sz_b = B.size // n
    
    C = A.reshape(sz_a, n) @ B.reshape(sz_b, n).T

    C /= n

    rshp = A.shape[:-len(B.shape)] + B.shape[:-1]
    return C.reshape(rshp)


def mean_outer(A, B):
    """
    Compute the means (along the first dimension) of the outer products of arrays A and B.  That is,
    if A is an xarray whose first dimension is pages and remaining dimensions are attributes, and
    B is a similar xarray with different attributes, the resulting array is the average (across pages)
    of the outer product of the fairness attributes.  This combines the two operations of intersectionally
    combining fairness attributes for a set of pages and computing the mean for an overall distribution,
    but it does so much more efficiently than the equivalent Xarray operations.

    It is the equivalent of the following code::

        (A * B).sum('page')
    """

    return xr.apply_ufunc(_mean_outer_ufunc, A, B, input_core_dims=[[A.dims[0]], [B.dims[0]]])