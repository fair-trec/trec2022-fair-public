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
# # Task 2 Evaluation
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

# %%
tbl_dir = Path('data/metric-tables/')

# %% [markdown]
# Set up progress bar and logging support:

# %% tags=[]
from tqdm.auto import tqdm
tqdm.pandas(leave=False)

# %% tags=[]
import sys, logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger('task2-eval')

# %% [markdown]
# Set up the RNG:

# %%
import seedbank
seedbank.initialize(20220101)
rng = seedbank.numpy_rng()

# %% [markdown]
# Import metric code:

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

# %%
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import wptrec.metrics as metrics
from wptrec.trecdata import scan_runs

# %% [markdown]
# And finally import the metric itself:

# %%
target = xr.open_dataarray(tbl_dir / f'task2-{DATA_MODE}-int-targets.nc')

# %%
metric = metrics.EELMetric(qrels.set_index('topic_id'), dimensions, target)

# %% [markdown]
# ## Importing Data
#
#

# %% [markdown]
# Let's load the runs now:

# %% tags=[]
runs = pd.DataFrame.from_records(row for rows in scan_runs(2, 'runs/2022') for row in rows)
runs

# %% tags=[]
runs.head()

# %% [markdown]
# ## Computing Metrics
#
# We are now ready to compute the metric for each (system,topic) pair.  Let's go!

# %% tags=[]
rank_exp = runs.groupby(['run_name', 'topic_id']).progress_apply(metric)
# rank_exp = rank_awrf.unstack()
rank_exp

# %% [markdown]
# Now let's average by runs:

# %% tags=[]
run_scores = rank_exp.groupby('run_name').mean()
run_scores


# %% [markdown]
# And bootstrap some confidence intervals:

# %%
def boot_ci(col, name='EE-L'):
    res = bootstrap([col], statistic=np.mean, random_state=rng)
    return pd.Series({
        f'{name}.SE': res.standard_error,
        f'{name}.Lo': res.confidence_interval.low,
        f'{name}.Hi': res.confidence_interval.high,
        f'{name}.W': res.confidence_interval.high - res.confidence_interval.low
    })


# %%
run_score_ci = rank_exp.groupby('run_name')['EE-L'].apply(boot_ci).unstack()
run_score_ci

# %%
run_score_full = run_scores.join(run_score_ci)
run_score_full.sort_values('EE-L', ascending=False, inplace=True)
run_score_full

# %% [markdown]
# ## Per-Topic Stats
#
# We need to return per-topic stats to each participant, at least for the score.

# %% tags=[]
topic_stats = rank_exp.groupby('topic_id').agg(['mean', 'median', 'min', 'max'])
topic_stats

# %% [markdown]
# Make final score analysis:

# %% tags=[]
topic_range = topic_stats.loc[:, 'EE-L']
topic_range = topic_range.drop(columns=['mean'])
topic_range

# %% [markdown]
# And now we combine scores with these results to return to participants.

# %% tags=[]
ret_dir = Path('results') / 'editors'
ret_dir.mkdir(exist_ok=True)
for system, s_runs in rank_exp.groupby('run_name'):
    aug = s_runs.join(topic_range).reset_index().drop(columns=['run_name'])
    fn = ret_dir / f'{system}.tsv'
    log.info('writing %s', fn)
    aug.to_csv(fn, sep='\t', index=False)

# %% [markdown]
# ## Charts and Relationships
#
# Now let's look at some overall distributions, etc.

# %%
sns.displot(x='EE-L', data=run_scores)

# %%
sns.scatterplot(x='EE-D', y='EE-R', data=run_scores)

# %% [markdown]
# ## Further Analysis
#
# That's how we capture the first-order analysis. Now let's look at things one dimension at a time.

# %% [markdown]
# ### Gender
#
# Let's just look at gender.

# %%
gender_tgt = pd.read_parquet(tbl_dir / f'task2-{DATA_MODE}-gender-target.parquet')
gender_tgt.head()

# %%
gender_tgt = xr.DataArray(gender_tgt.values, coords=[gender_tgt.index, gender_tgt.columns], dims=['topic_id', 'gender'])
gender_tgt

# %%
gender_metric = metrics.EELMetric(qrels.set_index('topic_id'), dimensions[2:3], gender_tgt)

# %%
rank_gender = runs.groupby(['run_name', 'topic_id']).progress_apply(gender_metric)
rank_gender

# %% [markdown]
# And aggregate:

# %%
rank_gender.groupby('run_name').mean()

# %% [markdown]
# ### Subject Geography
#
# And now subject geography.

# %%
sub_geo_tgt = pd.read_parquet(tbl_dir / f'task2-{DATA_MODE}-sub-geo-target.parquet')
sub_geo_tgt.head()

# %%
sub_geo_tgt = xr.DataArray(sub_geo_tgt.values, coords=[sub_geo_tgt.index, sub_geo_tgt.columns], dims=['topic_id', 'sub-geo'])
sub_geo_tgt

# %%
sub_geo_metric = metrics.EELMetric(qrels.set_index('topic_id'), [d for d in dimensions if d.name == 'sub-geo'], sub_geo_tgt)

# %%
rank_sub_geo = runs.groupby(['run_name', 'topic_id']).progress_apply(sub_geo_metric)
rank_sub_geo

# %% [markdown]
# And aggregate:

# %%
rank_sub_geo.groupby('run_name').mean()

# %%
