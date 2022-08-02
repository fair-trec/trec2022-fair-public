from pathlib import Path
from functools import reduce
import operator
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import jensenshannon

from .common import discount
from ..dimension import combine_alignments, mean_outer

class AWRFMetric:
    """
    AWRF metric implementation for Task 1.

    The metric is defined as the product of the nDCG and the AWRF.  This class stores the
    data structures needed to look up qrels and target alignments for each query.  It is
    callable, and usable directly as the function in a Pandas :meth:`pandas.DataFrameGroupBy.apply`
    call on a data frame that has been grouped by query identifier::

        run.groupby('qid')['page_id'].apply(metric)

    Since :meth:`__call__` extracts query ID directly from the name, if grouping by multiple
    fields, the query ID should be the **last** column.
    """

    def __init__(self, qrels, dimensions, qtgts, progress=lambda x, **kwargs: x):
        """
        Construct Task 1 metric.

        Args:
            qrels(pandas.DataFrame):
                The data frame of relevant documents, indexed by query ID.
            dimensions(list):
                A list of fairness dimensions.
            qtgts(xarray.DataArray):
                The target distribution, with page IDs on the first column.
        """
        self.qrels = qrels
        self.dimensions = dimensions
        self.qtgts = qtgts
        self.progress = progress

    def qr_ndcg(self, qrun, qrels, tgt_n=500):
        """
        Compute the per-ranking nDCG metric for Task 1.

        Args:
            qrun(array-like):
                The (ranked) document identifiers returned for a single query.
            qrels(array-like):
                The relevant document identifiers for this query.
            tgt_n(int):
                The maximum number of documents in a ranking.
        Returns:
            float:
                The nDCG score for the ranking.
        """
        if len(qrun) > tgt_n:
            raise ValueError(f'too many documents in query run')

        # length of run
        n = len(qrun)
        # max length of ideal run
        rel_n = min(tgt_n, len(qrels))

        # compute 1/0 utility scores
        util = np.isin(qrun, qrels).astype('f4')
        # compute discounts
        disc = discount(tgt_n)

        # compute nDCG
        run_dcg = np.sum(util * disc[:n])
        ideal_dcg = np.sum(disc[:rel_n])

        return run_dcg / ideal_dcg

    def qr_awrf(self, qrun, qtgt):
        """
        Compute the per-ranking AWRF metric for Task 1.

        Args:
            qrun(array-like):
                The (ranked) document identifiers returned for a single query.
            page_align(pandas.DataFrame):
                A Pandas data frame whose index is page IDs and columns are page alignment.
                values for each fairness category.
            qtgt(array-like):
                The target distribution for this query.
        Returns:
            float:
                The AWRF score for the ranking.
        """

        # length of run
        n = len(qrun)

        disc = discount(n)

        # work page by page for memory
        ralign = None
        for d, page in zip(self.progress(disc, leave=False), qrun):
            p_align = combine_alignments([
                d.page_align_xr.loc[page] for d in self.dimensions
            ])
            p_align *= d
            if ralign is None:
                ralign = p_align
            else:
                ralign += p_align

        ralign /= np.sum(disc)
        
        # now we have an alignment vector - compute distance
        dist = jensenshannon(ralign.values.ravel(), qtgt.values.ravel())
        
        # JS distance is sqrt of divergence
        return 1 - dist * dist
    
    def __call__(self, run, name=None):
        if name is None:
            name = run.name
        if isinstance(name, tuple):
            qid = name[-1]
        else:
            qid = name
        qrel = self.qrels.loc[qid]
        qtgt = self.qtgts.loc[qid]

        ndcg = self.qr_ndcg(run, qrel, 500)
        assert ndcg >= 0
        assert ndcg <= 1
        awrf = self.qr_awrf(run, qtgt)
        # assert awrf >= 0
        # assert awrf <= 1

        return pd.Series({
            'nDCG': ndcg,
            'AWRF': awrf,
            'Score': ndcg * awrf
        })