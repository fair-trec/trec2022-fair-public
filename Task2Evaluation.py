# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Task 2 Evaluation
#
# This notebook contains the evaluation for Task 1 of the TREC Fair Ranking track.

# %% [markdown]
# ## Setup
#
# We begin by loading necessary libraries:

# %% tags=[]
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import binpickle

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

# %%
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
# And finally import the metric itself:

# %%
metric = metrics.EELMetric.load('eval-qrels', 'task2-eval-int-targets', 'page-int-align', 'page-quality')

# %%
umetric = metrics.EUEMetric.load('eval-qrels', 'task2-eval-page-targets', 'page-int-align')

# %% [markdown]
# ## Importing Data
#
#

# %% [markdown]
# Let's load the runs now:

# %% tags=[]
runs = pd.DataFrame.from_records(row for (task, rows) in scan_runs('runs/2021') if task == 2 for row in rows)
runs

# %% tags=[]
runs.head()

# %% [markdown]
# We also need to load our topic eval data:

# %% tags=[]
topics = pd.read_json('data/eval-topics-with-qrels.json.gz', lines=True)
topics.head()

# %% [markdown]
# Tier 2 is the top 5 docs of the first 25 rankings.  Further, we didn't complete Tier 2 for all topics.

# %% tags=[]
t2_topics = topics.loc[topics['max_tier'] >= 2, 'id']

# %% tags=[]
r_top5 = runs['rank'] <= 5
r_first25 = runs['seq_no'] <= 25
r_done = runs['topic_id'].isin(t2_topics)
runs = runs[r_done & r_top5 & r_first25]
runs.info()

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
run_score_full

# %% [markdown]
# ## Underexposure Scores

# %%
from wptrec.metrics.under import qrs_page_exposure

# %%
rank_uexp = runs.groupby(['run_name', 'topic_id']).progress_apply(umetric)
rank_uexp

# %%
run_uscores = rank_uexp.groupby('run_name').mean()
run_uscores

# %% [markdown]
# ## Analyzing Scores
#
# What is the distribution of scores?

# %% tags=[]
run_scores.describe()

# %% tags=[]
sns.displot(x='EE-L', data=run_scores)
plt.savefig('figures/task1-eel-dist.pdf')
plt.show()

# %%
run_tbl_df = run_score_full[['EE-R', 'EE-D', 'EE-L']].copy()
run_tbl_df['EE-L 95% CI'] = run_score_full.apply(lambda r: "(%.3f, %.3f)" % (r['EE-L.Lo'], r['EE-L.Hi']), axis=1)
run_tbl_df

# %% tags=[]
run_tbl_df.sort_values('EE-L', ascending=True, inplace=True)
run_tbl_df

# %% tags=[]
run_tbl_fn = Path('figures/task2-runs.tex')
run_tbl = run_tbl_df.to_latex(float_format="%.4f", bold_rows=True, index_names=False)
run_tbl_fn.write_text(run_tbl)
print(run_tbl)

# %% tags=[]
sns.relplot(x='EE-R', y='EE-D', data=run_scores)
sns.rugplot(x='EE-R', y='EE-D', data=run_scores)
plt.savefig('figures/task2-eed-eer.pdf')
plt.show()

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
ret_dir = Path('results')
for system, runs in rank_exp.groupby('run_name'):
    aug = runs.join(topic_range).reset_index().drop(columns=['run_name'])
    fn = ret_dir / f'{system}.tsv'
    log.info('writing %s', fn)
    aug.to_csv(fn, sep='\t', index=False)

# %%
