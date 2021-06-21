"""
Implementation of metrics for TREC Fair Ranking 2021.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def qr_ndcg(qrun, qrels, tgt_n=100):
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
    disc = np.log2(np.arange(1, tgt_n + 1, dtype='f4'))
    disc[0] = 1  # reset log of first
    disc = np.reciprocal(disc)

    # compute nDCG
    run_dcg = np.sum(util * disc[:n])
    ideal_dcg = np.sum(disc[:rel_n])

    return run_dcg / ideal_dcg


def qr_awrf(qrun, page_align, qtgt):
    """
    Compute the per-ranking AWRF metric for Task 1.

    Args:
        qrun(array-like):
            The (ranked) document identifiers returned for a single query.
        page_align(pandas.Series):
            A Pandas series whose index is page IDs and columns are page alignment
            values for each fairness category.
        qtgt(array-like):
            The target distribution for this query.
    Returns:
        float:
            The AWRF score for the ranking.
    """

    # length of run
    n = len(qrun)

    disc = np.log2(np.arange(1, n + 1, dtype='f4'))
    disc[0] = 1  # reset log of first
    disc = np.reciprocal(disc)

    # get the page alignments
    ralign = page_align.loc[qrun]
    # discount and compute the weighted sum
    ralign = ralign.multiply(disc, axis=0)
    ralign = ralign.sum(axis=0) / np.sum(disc)
    # now we have an alignment vector
    dist = jensenshannon(ralign, qtgt)
    # JS distance is sqrt of divergence
    return 1 - dist * dist


class Task1Metric:
    """
    Task 1 metric implementation.

    The metric is defined as the product of the nDCG and the AWRF.  This class stores the
    data structures needed to look up qrels and target alignments for each query.  It is
    callable, and usable directly as the function in a Pandas :meth:`pandas.DataFrameGroupBy.apply`
    call on a data frame that has been grouped by query identifier::

        run.groupby('qid')['doc_id'].apply(metric)

    Since :meth:`__call__` extracts query ID directly from the name, this doesn't work if you
    have a frame that you are grouping by more than one column.
    """

    def __init__(self, qrels, page_align, qtgts):
        """
        Construct Task 1 metric.

        Args:
            qrels(pandas.DataFrame):
                The data frame of relevant documents, indexed by query ID.
            page_align(pandas.DataFrame):
                The data frame of page alignments for fairness criteria, indexed by page ID.
            qtgts(pandas.DataFrame):
                The data frame of query target distributions, indexed by query ID.
        """
        self.qrels = qrels
        self.page_align = page_align
        self.qtgts = qtgts
    
    def __call__(self, run):
        qid = run.name
        qrel = self.qrels.loc[qid]
        qtgt = self.qtgts.loc[qid]

        ndcg = qr_ndcg(run, qrel, 1000)
        assert ndcg >= 0
        assert ndcg <= 1
        awrf = qr_awrf(run, self.page_align, qtgt)
        assert awrf >= 0
        assert awrf <= 1

        return pd.Series({
            'nDCG': ndcg,
            'AWRF': awrf,
            'Score': ndcg * awrf
        })
