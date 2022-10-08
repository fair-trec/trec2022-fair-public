"""
Logic for working with fairness definitions.
"""
from array import array
from functools import reduce
import operator as op
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

    def alignments(self, pages):
        "Get alignments for a page."
        align = self.page_align_xr.reindex({'page': pages}).copy()
        missing = align.isnull().all(self.name)
        if not missing.any():
            return align

        assert missing.dims == ('page',)
        coords = self.page_align_xr.coords[self.name]
        row = xr.DataArray(np.zeros(len(coords)), coords={self.name: coords})
        if self.has_unknown:
            row.loc['@UNKNOWN'] = 1
        else:
            row[:] = 1 / len(row)
        
        align[missing] = row
        return align

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


def agg_alignments(arrays, agg='sum', weights=None):
    """
    Aggregate the alignments for a set of keys, into a single aggregated alignment matrix.  This
    works with {func}`combine_alignments`, {func}`mean_outer`, and {func}`sum_outer` to combine
    the selected matrices.

    Args:
        arrays(list of xarray.DataArray):
            List of the alignment matrices for each attribute.  The first dimension of each
            array is assumed to correspond to ``keys`` (e.g. page IDs).
        agg(str):
            The aggregate to perform.  Valid options are ``sum`` and ``mean``.
        weights(numpy.ndarray or None):
            Weights to apply to each element.  If provided, must be the same length as the
            first dimension of the arrays.
    """
    assert agg in ('mean', 'sum')

    npages = arrays[0].shape[0]
    assert all([a.shape[0] == npages for a in arrays])
    assert all([len(a.shape) == 2 for a in arrays])
    if weights is not None:
        assert weights.shape == (npages,)

    sizes = [a.shape[1] for a in arrays]
    
    ##############
    # STEP 1: build two arrays with roughly balanced mass
    
    if len(sizes) > 2:
        # compute mass (in log space, so we have nice sums)
        log_mass = np.log(np.array(sizes))
        tot_mass = np.sum(log_mass)
        cum_mass = np.cumsum(log_mass)
        # how many do we have to get half the mass?
        n_less = np.sum(cum_mass / tot_mass < 0.5)
        # split our arrays into these two lengths
        left = combine_alignments(arrays[:n_less])
        assert left.shape[0] == npages
        right = combine_alignments(arrays[n_less:])
        assert right.shape[0] == npages
    elif len(sizes) == 2:
        left, right = arrays
    else:
        left = arrays[0]
        right = None

    #############
    # STEP 2: apply weights, if we have them; we only need to multiply one matrix
    if weights is not None:
        wa = xr.DataArray(weights, coords=[left.coords[left.dims[0]]], dims=left.dims[0])
        left *= wa
    
    #############
    # STEP 3: aggregate the outer products of the balanced arrays and return
    if weights is None:
        if agg == 'sum':
            return sum_outer(left, right)
        elif agg == 'mean':
            return mean_outer(left, right)
    else:
        sums = sum_outer(left, right)
        if agg == 'mean':
            sums /= np.sum(weights)
        return sums


def _mean_outer_ufunc(A, B):
    """
    Worker function for {func}`mean_outer`.
    """

    # Since this is called with the page in the input core dimensions, the page
    # is moved to the **last** dimension of the arrays passed to this function.
    n = A.shape[-1]
    sz_a = A.size // n
    sz_b = B.size // n
    
    C = A.reshape(sz_a, n) @ B.reshape(sz_b, n).T

    C /= n

    rshp = A.shape[:-len(B.shape)] + B.shape[:-1]
    return C.reshape(rshp)


def _sum_outer_ufunc(A, B):
    # Since this is called with the page in the input core dimensions, the page
    # is moved to the **last** dimension of the arrays passed to this function.
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

        (A * B).mean('page')
    """

    if B is None:
        return A.mean(axis=0)
    else:
        return xr.apply_ufunc(_mean_outer_ufunc, A, B, input_core_dims=[[A.dims[0]], [B.dims[0]]])

def sum_outer(A, B):
    """
    Compute the sums (along the first dimension) of the outer products of arrays A and B.  That is,
    if A is an xarray whose first dimension is pages and remaining dimensions are attributes, and
    B is a similar xarray with different attributes, the resulting array is the average (across pages)
    of the outer product of the fairness attributes.  This combines the two operations of intersectionally
    combining fairness attributes for a set of pages and computing the mean for an overall distribution,
    but it does so much more efficiently than the equivalent Xarray operations.

    It is the equivalent of the following code::

        (A * B).sum('page')
    """

    if B is None:
        return A.sum(axis=0)
    else:
        return xr.apply_ufunc(_sum_outer_ufunc, A, B, input_core_dims=[[A.dims[0]], [B.dims[0]]])