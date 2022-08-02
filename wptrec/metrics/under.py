from pathlib import Path
import numpy as np
import pandas as pd

from .common import discount, work_order

def qr_page_exposure(qrun):
    """
    Compute the group exposure from a single ranking.

    Args:
        qrun(array-like):
            The (ranked) document identifiers returned for a single query.
    Returns:
        numpy.ndarray:
            The group exposures for the ranking (unnormalized)
    """

    # length of run
    n = len(qrun)
    disc = discount(n)

    # return the page exposures
    res = pd.Series(disc, index=qrun)
    res.index.name = 'page'

    return res


def qrs_page_exposure(qruns):
    """
    Compute the group exposure from a sequence of rankings.

    Args:
        qruns(array-like):
            A data frame of document identifiers for a single query.
    Returns:
        pandas.Series:
            Each group's expected exposure for the ranking.
    """

    rexp = qruns.groupby('seq_no')['page_id'].apply(qr_page_exposure)
    exp = rexp.unstack().fillna(0).mean(axis=0)
    assert exp.index.name == 'page'
    exp.name = 'exposure'

    return exp


class EUEMetric:
    """
    Task 2 underexposure metric implementation.

    This class stores the data structures needed to look up qrels and target exposures for each
    query and document.  It is callable, and usable directly as the function in a Pandas
    :meth:`pandas.DataFrameGroupBy.apply` call on a data frame that has been grouped by query
    identifier::

        run.groupby('qid').apply(metric)

    Since :meth:`__call__` extracts query ID directly from the name, this doesn't work if you have a
    frame that you are grouping by more than one column.
    """

    def __init__(self, qrels, page_align, qptgts):
        """
        Construct Task 2 metric.

        Args:
            qrels(pandas.DataFrame):
                The data frame of relevant documents, indexed by query ID.
            page_align(pandas.DataFrame):
                The data frame of page alignments for fairness criteria, indexed by page ID.
            qptgts(pandas.DataFrame):
                The data frame of query page target exposures, indexed by query ID and page ID.
        """
        self.qrels = qrels
        self.page_align = page_align
        self.qptgts = qptgts

    @classmethod
    def load(cls, qrels, qptgts, page_align, dir='data/metric-tables'):
        dir = Path(dir)
        qrel_df = pd.read_csv(dir / f'{qrels}.csv.gz', index_col='id')
        qptgt_df = pd.read_csv(dir / f'{qptgts}.csv.gz').set_index(['id', 'page'])['target']
        pa_df = pd.read_csv(dir / f'{page_align}.csv.gz', index_col='page')
        return cls(qrel_df, pa_df, qptgt_df)
    
    def __call__(self, sequence):
        if isinstance(sequence.name, tuple):
            qid = sequence.name[-1]
        else:
            qid = sequence.name
        
        # get the target exposure
        qtgt = self.qptgts.loc[qid]
        assert qtgt.index.name == 'page'

        # get the system exposure
        s_exp = qrs_page_exposure(sequence)
        
        # normalize exposures
        qtgt = qtgt / np.sum(qtgt)
        s_exp = s_exp / np.sum(s_exp)

        qtgt, s_exp = qtgt.align(s_exp, fill_value=0)

        # compute the underexposure
        loss = qtgt - s_exp  # positive for undexposure
        uexp = np.maximum(loss, 0.0)

        uexp = uexp.to_frame('under')
        
        # group-aggregate the underexposure
        qpa, uexp = self.page_align.align(uexp, join='inner', axis=0)
        g_uexp = (uexp.T @ qpa).T['under']

        return pd.Series({
            'L2UEXP': np.dot(g_uexp, g_uexp),
            'TotUEXP': np.sum(g_uexp),
        })