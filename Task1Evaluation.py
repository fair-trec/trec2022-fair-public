# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Task 1 Evaluation
#
# This notebook contains the evaluation for Task 1 of the TREC Fair Ranking track.

# %% tags=["parameters"]
DATA_MODE = 'eval'

# %% tags=[]
import wptrec
wptrec.DATA_MODE = DATA_MODE

# %% [markdown]
# ## Setup
#
# We begin by loading necessary libraries:

# %% tags=[]
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import binpickle

# %% tags=[]
tbl_dir = Path('data/metric-tables')

# %% [markdown]
# Set up progress bar and logging support:

# %% tags=[]
from tqdm.auto import tqdm
tqdm.pandas(leave=False)

# %% tags=[]
import sys, logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger('task1-eval')

# %% [markdown]
# Set up the RNG:

# %% tags=[]
import seedbank
seedbank.initialize(20220101)
rng = seedbank.numpy_rng()

# %% [markdown]
# Import metric code:

# %%
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import wptrec.metrics as metrics
from wptrec.trecdata import scan_runs

# %% [markdown]
# And finally import the metric itself.  For Task 1, this uses:
#
# * evaluation qrels
# * evaluation intersectional targets
# * all dimensions (with their page alignments)

# %% tags=[]
from MetricInputs import qrels, dimensions

# %% tags=[]
target = xr.open_dataarray(tbl_dir / f'task1-{DATA_MODE}-int-targets.nc')

# %% tags=[]
metric = metrics.AWRFMetric(qrels.set_index('topic_id'), dimensions, target)

# %% [markdown]
# ## Importing Data

# %% [markdown]
# Let's load the runs now:

# %% tags=[]
runs = pd.DataFrame.from_records(row for rows in scan_runs(1, 'runs/2022') for row in rows)
runs

# %% [markdown] tags=[]
# ## Computing Metrics
#
# We are now ready to compute the metric for each (system,topic) pair.  Let's go!

# %% tags=[]
rank_awrf = runs.groupby(['run_name', 'topic_id'])['page_id'].progress_apply(metric)
rank_awrf = rank_awrf.unstack()
rank_awrf

# %% [markdown]
# Make sure we aren't missing anything:

# %% tags=[]
rank_awrf[rank_awrf['Score'].isnull()]

# %% [markdown]
# Now let's average by runs:

# %% tags=[]
run_scores = rank_awrf.groupby('run_name').mean()
run_scores.sort_values('Score', ascending=False, inplace=True)
run_scores


# %% [markdown]
# And bootstrap some confidence intervals:

# %% tags=[]
def boot_ci(col):
    res = bootstrap([col], statistic=np.mean, random_state=rng)
    return pd.Series({
        'Score.SE': res.standard_error,
        'Score.Lo': res.confidence_interval.low,
        'Score.Hi': res.confidence_interval.high,
        'Score.W': res.confidence_interval.high - res.confidence_interval.low
    })


# %% tags=[]
run_score_ci = rank_awrf.groupby('run_name')['Score'].apply(boot_ci).unstack()
run_score_ci

# %% tags=[]
run_score_full = run_scores.join(run_score_ci)
run_score_full

# %% tags=[]
run_tbl_df = run_score_full[['nDCG', 'AWRF', 'Score']].copy()
run_tbl_df['95% CI'] = run_score_full.apply(lambda r: "(%.3f, %.3f)" % (r['Score.Lo'], r['Score.Hi']), axis=1)
run_tbl_df

# %% [markdown]
# Combine them:

# %% tags=[]
run_tbl_fn = Path('figures/task1-runs.tex')
run_tbl = run_tbl_df.to_latex(float_format="%.4f", bold_rows=True, index_names=False)
run_tbl_fn.write_text(run_tbl)
print(run_tbl)

# %% [markdown]
# ## Analyzing Scores
#
# What is the distribution of scores?

# %% tags=[]
run_scores.describe()

# %% tags=[]
sns.displot(x='Score', data=run_scores)
plt.savefig('figures/task1-score-dist.pdf')
plt.show()

# %% tags=[]
sns.relplot(x='nDCG', y='AWRF', data=run_scores)
sns.rugplot(x='nDCG', y='AWRF', data=run_scores)
plt.savefig('figures/task1-ndcg-awrf.pdf')
plt.show()

# %% [markdown]
# ## Per-Topic Stats
#
# We need to return per-topic stats to each participant, at least for the score.

# %% tags=[]
topic_stats = rank_awrf.groupby('topic_id').agg(['mean', 'median', 'min', 'max'])
topic_stats

# %% [markdown]
# Make final score analysis:

# %% tags=[]
topic_range = topic_stats.loc[:, 'Score']
topic_range = topic_range.drop(columns=['mean'])
topic_range

# %% [markdown]
# And now we combine scores with these results to return to participants.

# %% tags=[]
ret_dir = Path('results') / 'coordinators'
ret_dir.mkdir(exist_ok=True)
for system, runs in rank_awrf.groupby('run_name'):
    aug = runs.join(topic_range).reset_index().drop(columns=['run_name'])
    fn = ret_dir / f'{system}.tsv'
    log.info('writing %s', fn)
    aug.to_csv(fn, sep='\t', index=False)

# %%
