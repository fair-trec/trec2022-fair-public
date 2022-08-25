"""
Produce TREC fair ranking runs from an oracle.

This script assumes the data lives in a directory 'data'.  It loads the training
topics and uses them as an oracle for producing rankings.

Note that the unfairness support is *extremely slow* and not yet debugged.

Usage:
    oracle-runs.py --task1 [options]
    oracle-runs.py --task2 [options]

Options:
    -v, --verbose
        Write verbose logging output.
    -o FILE
        Write output to FILE.
    -p PREC, --precision=PREC
        Produce results with the specified precision [default: 0.9].
    -U MIX, --unfairness=MIX
        Produce results with the specified "unfairness" [default: 0.0].
        Only works for task 2.
    --task1
        Create runs for Task 1.
    --task2
        Create runs for Task 2 (sequences).
    --eval-topics
        Use eval topics instead of training topics.
"""

import sys
from pathlib import Path
import logging
from tqdm import tqdm
from docopt import docopt

import xarray as xr
import pandas as pd
import numpy as np
from scipy.special import softmax

from wptrec.dimension import agg_alignments, combine_alignments

_log = logging.getLogger('oracle-runs')


def load_metadata():
    meta_f = Path('data/trec_2022_articles.parquet')
    _log.info('reading %s', meta_f)
    meta = pd.read_parquet(meta_f, columns=['first_letter'])
    # make sure we have the right index
    assert meta.index.name == 'page_id'
    return meta


def load_topics(opts):
    key = 'eval' if opts['--eval-topics'] else 'train'
    topic_f = Path(f'data/trec_2022_{key}_reldocs.jsonl')
    _log.info('reading %s', topic_f)
    topics = pd.read_json(topic_f, lines=True)
    return topics


def load_alignments(opts):
    from MetricInputs import dimensions
    return [d.page_align_xr for d in dimensions]


def sample_docs(rng, meta, rel, n, prec, weights=None):
    _log.debug('sampling %d rel items (n=%d, prec=%.2f)', len(rel), n, prec)
    n_rel = min(int(n * prec), len(rel))
    n_unrel = n - n_rel
    
    rel = np.array(rel)
    all = pd.Series(meta.index)
    unrel = all[~all.isin(rel)].values

    samp_rel = rng.choice(rel, n_rel, replace=False, p=weights)
    samp_unrel = rng.choice(unrel, n_unrel, replace=False)
    samp = np.concatenate([samp_rel, samp_unrel])
    rng.shuffle(samp)
    return pd.Series(samp)


def weigh_docs(docs, aligns, unfairness):
    docs = docs.values
    # extract per-doc alignments
    aligns = [a.loc[docs] for a in aligns]

    # compute overall alignment probabilities
    p_a = agg_alignments(aligns, 'mean')
    
    # compute overall doc probabilities
    p_d = 1.0 / len(docs)

    # compute extremified concentrations
    conc_p = softmax(p_a * 10000)
    conc_p *= unfairness
    uf_p_a = p_a * (1-unfairness) + conc_p

    def dw(doc):
        das = [a.loc[doc] * p_d / p_a for a in aligns]
        das = combine_alignments(das)
        das *= uf_p_a
        return das.sum()

    # compute individual document probabilities
    doc_weights = [
        dw(doc) for doc in tqdm(docs, desc='docs', leave=False)
    ]
    return pd.Series(docs, doc_weights)


def task1_run(opts, meta, topics):
    rng = np.random.default_rng()
    rank_len = 500
    prec = float(opts['--precision'])

    rels = topics[['id', 'rel_docs']].set_index('id').explode('rel_docs')

    def sample(df):
        return sample_docs(rng, meta, df['rel_docs'], rank_len, prec)
    
    runs = rels.groupby('id').progress_apply(sample)
    runs.columns.name = 'rank'
    runs = runs.stack().reset_index(name='page_id')
    _log.info('sample runs:\n%s', runs)
    return runs[['id', 'page_id']]


def task2_run(opts, meta, topics):
    rng = np.random.default_rng()
    rank_len = 20
    run_count = 100
    prec = float(opts['--precision'])
    unf = float(opts['--unfairness'])

    rels = topics[['id', 'rel_docs']].set_index('id').explode('rel_docs')
    
    if unf > 0:
        _log.info('unfairness parameter: %.3f', unf)
        dims = load_alignments(opts)
    else:
        dims = None

    def one_sample(df, weights):
        return sample_docs(rng, meta, df['rel_docs'], rank_len, prec, weights)
    
    def multi_sample(df):
        weights = None if dims is None else weigh_docs(df['rel_docs'], dims, unf)
        runs = dict((i+1, one_sample(df, weights)) for i in tqdm(range(run_count), 'reps', leave=False))
        rdf = pd.DataFrame(runs)
        rdf.columns.name = 'seq_no'
        rdf.index.name = 'rank'
        return rdf.T
    
    runs = rels.groupby('id').progress_apply(multi_sample)
    runs = runs.stack().reset_index(name='page_id')
    _log.info('multi-sample runs:\n%s', runs)
    return runs[['id', 'seq_no', 'page_id']]


def main(opts):
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)
    tqdm.pandas(leave=False)

    meta = load_metadata()
    topics = load_topics(opts)

    if opts['--task1']:
        runs = task1_run(opts, meta, topics)
        dft_out = 'runs/2022/task1.tsv'
    elif opts['--task2']:
        runs = task2_run(opts, meta, topics)
        dft_out = 'runs/2022/task2.tsv'
    else:
        raise ValueError('no task specified')
    
    out_file = opts.get('-o', dft_out)
    _log.info('writing to %s', out_file)
    runs.to_csv(out_file, index=False, sep='\t')


if __name__ == '__main__':
    opts = docopt(__doc__)
    main(opts)