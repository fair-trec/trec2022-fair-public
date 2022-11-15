from pathlib import Path
from re import A
import numpy as np
import pandas as pd
import xarray as xr

from .common import discount, work_order
from ..dimension import agg_alignments


def _norm(x, lb, ub):
    return (x - lb) / (ub - lb)


class EELMetric:
    """
    Task 2 metric implementation.

    This class stores the data structures needed to look up qrels and target exposures for each
    query.  It is callable, and usable directly as the function in a Pandas
    :meth:`pandas.DataFrameGroupBy.apply` call on a data frame that has been grouped by query
    identifier::

        run.groupby('qid').apply(metric)

    Normalization, if enabled, is based https://github.com/fair-trec/fair-trec-tools/blob/master/eval/metrics.py.
    """

    def __init__(self, qrels, dimensions, qtgts, only_rel=True, target_len=20, normalize=False):
        """
        Construct Task 2 metric.

        Args:
            qrels(pandas.DataFrame):
                The data frame of relevant documents, indexed by query ID.
            dimensions(list):
                A list of fairness dimensions.
            qtgts(xarray.DataArray):
                The target distribution, with topic IDs on the first dimension.
            only_rel(bool):
                Whether to only allow relevant documents, or all documents, to contribute
                to system exposure.
            target_len(int):
                The number of items each ranking should have.
            normalize(bool):
                Whether to attempt to normalize the metric scores.
                CURRENTLY BROKEN, DO NOT USE.
        """
        self.qrels = qrels
        self.dimensions = dimensions
        self.qtgts = qtgts
        self.only_rel = only_rel
        self.target_len = target_len
        self.normalize = normalize

    def qrs_exposure(self, qruns: pd.DataFrame, qrels):
        """
        Compute the unnormalized group exposure from a sequence of rankings.

        Args:
            qruns(array-like):
                A data frame of document identifiers for a single query.
            page_align(pandas.DataFrame):
                A Pandas data frame whose index is page IDs and columns are page alignment
                values for each fairness category.
        Returns:
            pandas.Series:
                Each group's expected exposure for the ranking.
        """

        # get the relevance data
        rel_col = qruns['page_id'].isin(qrels['page_id'])
        
        # add to data frame & get position exposures
        qrw = qruns.assign(is_rel=rel_col)
        qrw = qrw.groupby('seq_no').apply(lambda df: pd.DataFrame({
            'page_id': df['page_id'],
            'is_rel': df['is_rel'],
            'discount': discount(len(df)),
        }).reset_index(drop=True))

        if not qrw['is_rel'].any():
            # there is no relevant data - zero exposure
            dims = self.qtgts.dims[1:]
            zed = xr.zeros_like(self.qtgts[0])
            del zed.coords['topic_id']
            return zed

        # we only count relevant documents (as per EE paper)
        # this is a *change* from the 2021 Task 2 metric implementation!
        if self.only_rel:
            qrw = qrw[qrw['is_rel']]

        # we compute system exposure as the discount-weighted mean of the
        # page alignment matrices.
        arrays = [
            d.page_align_xr.loc[qrw['page_id'].values].astype('float64') for d in self.dimensions
        ]
        # we sum *within* a ranking, and average *across* rankings.
        exp = agg_alignments(arrays, 'sum', qrw['discount'].values)
        exp /= qruns['seq_no'].nunique()

        return exp
    
    def __call__(self, sequence, *, normalize=False, details=False):
        normalize = normalize or self.normalize
        if isinstance(sequence.name, tuple):
            qid = sequence.name[-1]
        else:
            qid = sequence.name
        tgt_exp = self.qtgts.loc[qid]

        qrels = self.qrels.loc[qid]
        s_exp = self.qrs_exposure(sequence, qrels)
        
        # tgt exposure is on a per-exposure-unit basis - scale up based on sequence len
        # get the weights for a full-length sequence
        tgt_weights = discount(self.target_len)
        # compute the total exposure weight for a full-length sequence
        avail_exp = np.sum(tgt_weights)
        # and scale up the target exposures
        tgt_exp = tgt_exp * avail_exp
        
        # compute the delta
        delta = s_exp - tgt_exp

        # prepare for computations - unravell multiple dimensions
        s_exp = np.ravel(s_exp)
        tgt_exp = np.ravel(tgt_exp)
        delta = np.ravel(delta)
        n_dims = len(s_exp)

        # compute metric values
        ee_disp = np.dot(s_exp, s_exp)
        ee_rel = np.dot(s_exp, tgt_exp)
        ee_loss = np.dot(delta, delta)

        res = None

        if normalize:
            res = {
                'EE-L-raw': ee_loss,
                'EE-D-raw': ee_disp,
                'EE-R-raw': ee_rel,
            }

            ## EE-D
            # Lower bound: all available exposure equally distributed
            ee_d_lb = np.square(avail_exp / n_dims) * n_dims

            # Upper bound: all exposure concentrated on one element
            ee_d_ub = np.square(avail_exp)

            ee_disp = _norm(ee_disp, ee_d_lb, ee_d_ub)
            res.update({
                'EE-D': ee_disp,
                'EE-D-lb': ee_d_lb,
                'EE-D-ub': ee_d_ub,
            })

            ## EE-R
            # Lower bound: all exposure to non-relevant groups
            # loose, since we don't have any fully non-relevant groups
            ee_r_lb = 0

            # Upper bound: exposure perfectly aligned with relevance
            ee_r_ub = np.dot(tgt_exp, tgt_exp)

            ee_rel = _norm(ee_rel, ee_r_lb, ee_r_ub)

            res.update({
                'EE-R': ee_rel,
                'EE-R-lb': ee_r_lb,
                'EE-R-ub': ee_r_ub,
            })

            ## EE-L
            # Lower bound: no loss
            ee_l_lb = 0

            # Upper bound (loose): put all exposure on a group w/ infinitesimal exposure
            # this is because moving any mass from that group another would reduce
            # both squared differences, reducing overall loss
            ee_l_ub = ee_r_ub + ee_d_ub

            ee_loss = _norm(ee_loss, ee_l_lb, ee_l_ub)

            res.update({
                'EE-L': ee_loss,
                'EE-L-lb': ee_l_lb,
                'EE-L-ub': ee_l_ub,
            })
    
        if res is None or not details:
            res = {
                'EE-L': ee_loss,
                'EE-D': ee_disp,
                'EE-R': ee_rel,
            }
        
        return pd.Series(res)
        