from pathlib import Path
import numpy as np
import pandas as pd

from .common import discount, work_order

def qr_exposure(qrun, page_align):
    """
    Compute the group exposure from a single ranking.

    Args:
        qrun(array-like):
            The (ranked) document identifiers returned for a single query.
        page_align(pandas.DataFrame):
            A Pandas data frame whose index is page IDs and columns are page alignment
            values for each fairness category.
    Returns:
        numpy.ndarray:
            The group exposures for the ranking (unnormalized)
    """

    # length of run
    n = len(qrun)
    disc = discount(n)

    # get the page alignments
    ralign = page_align.reindex(qrun)
    # discount and compute the weighted sum
    ralign = ralign.multiply(disc, axis=0)
    ralign = ralign.sum(axis=0)

    assert len(ralign) == page_align.shape[1]

    return ralign


def qrs_exposure(qruns, page_align):
    """
    Compute the group exposure from a sequence of rankings.

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

    rexp = qruns.groupby('seq_no')['page_id'].apply(qr_exposure, page_align=page_align)
    exp = rexp.unstack().fillna(0).mean(axis=0)
    assert len(exp) == page_align.shape[1]

    return exp


def qw_tgt_exposure(qw_counts: pd.Series) -> pd.Series:
    """
    Compute the target exposure for each work level for a query.

    Args:
        qw_counts(pandas.Series):
            The number of articles the query has for each work level.
    
    Returns:
        pandas.Series:
            The per-article target exposure for each work level.
    """
    if 'id' == qw_counts.index.names[0]:
        qw_counts = qw_counts.reset_index(level='id', drop=True)
    qwc = qw_counts.reindex(work_order, fill_value=0).astype('i4')
    tot = int(qwc.sum())
    da = discount(tot)
    qwp = qwc.shift(1, fill_value=0)
    qwc_s = qwc.cumsum()
    qwp_s = qwp.cumsum()
    res = pd.Series(
        [np.mean(da[s:e]) for (s, e) in zip(qwp_s, qwc_s)],
        index=qwc.index
    )
    return res


class EELMetric:
    """
    Task 2 metric implementation.

    This class stores the data structures needed to look up qrels and target exposures for each
    query.  It is callable, and usable directly as the function in a Pandas
    :meth:`pandas.DataFrameGroupBy.apply` call on a data frame that has been grouped by query
    identifier::

        run.groupby('qid').apply(metric)

    Since :meth:`__call__` extracts query ID directly from the name, this doesn't work if you have a
    frame that you are grouping by more than one column.
    """

    def __init__(self, qrels, page_align, page_work, qtgts):
        """
        Construct Task 2 metric.

        Args:
            qrels(pandas.DataFrame):
                The data frame of relevant documents, indexed by query ID.
            page_align(pandas.DataFrame):
                The data frame of page alignments for fairness criteria, indexed by page ID.
            qtgts(pandas.DataFrame):
                The data frame of query target exposures, indexed by query ID.
        """
        self.qrels = qrels
        self.page_align = page_align
        self.page_work = page_work
        self.qtgts = qtgts

    @classmethod
    def load(cls, qrels, qtgts, page_align, page_work, dir='data/metric-tables'):
        dir = Path(dir)
        qrel_df = pd.read_csv(dir / f'{qrels}.csv.gz', index_col='id')
        qtgt_df = pd.read_csv(dir / f'{qtgts}.csv.gz', index_col='id')
        pa_df = pd.read_csv(dir / f'{page_align}.csv.gz', index_col='page')
        pw_df = pd.read_parquet(dir / f'{page_work}.parquet')
        return cls(qrel_df, pa_df, pw_df, qtgt_df)
    
    def __call__(self, sequence):
        if isinstance(sequence.name, tuple):
            qid = sequence.name[-1]
        else:
            qid = sequence.name
        qtgt = self.qtgts.loc[qid]

        s_exp = qrs_exposure(sequence, self.page_align)
        avail_exp = np.sum(discount(50))
        tgt_exp = qtgt * avail_exp
        delta = s_exp - tgt_exp

        ee_disp = np.dot(s_exp, s_exp)
        ee_rel = np.dot(s_exp, tgt_exp)
        ee_loss = np.dot(delta, delta)

        return pd.Series({
            'EE-L': ee_loss,
            'EE-D': ee_disp,
            'EE-R': ee_rel,
        })