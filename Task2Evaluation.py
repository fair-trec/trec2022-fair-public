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
from functools import reduce
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
from wptrec.trecdata import scan_runs, scan_teams

# %% [markdown]
# And finally import the metric itself.  For Task 1, this uses:
#
# * evaluation qrels
# * evaluation intersectional targets
# * all dimensions (with their page alignments)

# %% tags=[]
from MetricInputs import qrels, dimensions

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

# %% tags=[]
runs.head()

# %% [markdown]
# And the teams:

# %%
team_runs = scan_teams('runs/2022')
team_runs

# %%
run_team = team_runs.set_index('run')

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
run_scores= run_scores.join(run_team)
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

# %%
run_tbl_df = run_score_full[['EE-R', 'EE-D', 'EE-L']].copy()
run_tbl_df['EE-L 95% CI'] = run_score_full.apply(lambda r: (r['EE-L.Lo'], r['EE-L.Hi']), axis=1)
run_tbl_df

# %%
run_tbl_df.sort_values('EE-L', ascending=True, inplace=True)
column_styles = {
    'EE-R': lambda x: r'%0.4f' % (x,),
    'EE-L': lambda x: r'%0.4f' % (x,),
    'EE-D': lambda x: r'%0.4f' % (x,),
    'EE-L 95% CI': lambda x: r'(%0.4f, %0.4f)' % (x[0], x[1]),
}
dfs = run_tbl_df.style.format(column_styles)
dfs = dfs.highlight_min('EE-L', props='font-weight: bold')
dfs = dfs.highlight_min('EE-D', props='font-weight: bold')
dfs = dfs.highlight_max('EE-R', props='font-weight: bold')
dfs

# %%
run_tbl_fn = Path('figures/task2-runs.tex')
column_styles = {
    'EE-R': lambda x: r'%0.4f' % (x,),
    'EE-L': lambda x: r'%0.4f' % (x,),
    'EE-D': lambda x: r'%0.4f' % (x,),
    'EE-L 95% CI': lambda x: r'(%0.4f, %0.4f)' % (x[0], x[1]),
}
style = run_tbl_df.style.format(column_styles)
style = style.format_index(lambda x: r'\textbf{%s}' % (x,), escape='latex')
style = style.format_index(axis=1, escape='latex')
style = style.hide(names=True)
run_tbl = style.to_latex()
run_tbl_fn.write_text(run_tbl)
print(run_tbl)

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
sns.scatterplot(x='EE-D', y='EE-R', hue='team', data=run_scores)
sns.rugplot(x='EE-D', y='EE-R', data=run_scores)
plt.savefig('figures/task2-eed-eer.pdf')
plt.show()

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

# %% [markdown]
# ## Single-Attribute Performance
#
# Having poked at a few individual attributes, we're now going to run each attribute as a single attribute.

# %%
res1d_d = {}
dim_loop = tqdm(dimensions, desc='dims', leave=False)
for dim in dim_loop:
    dim_loop.set_postfix_str(dim.name)
    t1d = pd.read_parquet(tbl_dir / f'task2-{DATA_MODE}-{dim.name}-target.parquet').astype('float64')
    t1d = xr.DataArray(t1d, dims=['topic_id', dim.name])
    m1d = metrics.EELMetric(qrels.set_index('topic_id'), [dim], t1d)
    res1d_d[dim.name] = runs.groupby(['run_name', 'topic_id']).progress_apply(m1d)

# %%
res1d = pd.concat(res1d_d, names=['dim'])
res1d = res1d.reset_index()
res1d

# %% [markdown]
# Now let's group things to get per-dimension metrics!

# %%
rr_1d = res1d.groupby(['dim', 'run_name'])['EE-L'].mean()
rr_1d = rr_1d.unstack('dim')
rr_1d

# %%
df_1d = run_scores[['EE-L']].rename(columns={'EE-L': 'Overall'}).join(rr_1d)
df_1d.sort_values('Overall', inplace=True)
df_fmt = df_1d.style.highlight_min(props='font-weight: bold').format(lambda x: '%.3f' % (x,))
df_fmt

# %%
df_1d_fn = Path('figures/task2-single.tex')
style = df_1d.style.highlight_min(props='font: bold;').format(lambda x: '%.3f' % (x,))
style = style.format_index(axis=0, escape='latex')
style = style.hide(names=True)
df_tex = style.to_latex()
df_1d_fn.write_text(df_tex)
print(df_tex)


# %% [markdown]
# ## Attribute Subset Performance
#
# We also want to look at the peformance over *subsets* of the original attributes.  For this, we need two pieces:
#
# - The dimensions
# - The reduced target
#
# We'll get the reduced target by marginalizing.  Let's make a function to get dimensions and reduced targets:

# %%
def subset_dims(dims):
    return [d for d in dimensions if d.name in dims]


# %%
def subset_tgt(dims):
    names = [d.name for d in dimensions if d.name not in dims]
    return target.sum(names)


# %% [markdown]
# And a demo:

# %%
subset_tgt(['sub-geo', 'gender'])

# %% [markdown]
# Great!  Now we're ready to go!

# %% [markdown]
# ### Gender and Geography
#
# Last year, we used subject geography and gender.  Let's generate metric results from those.

# %%
geo_gender_metric = metrics.EELMetric(qrels.set_index('topic_id'), subset_dims(['sub-geo', 'gender']), subset_tgt(['sub-geo', 'gender']))

# %%
geo_gender_res = runs.groupby(['run_name', 'topic_id']).progress_apply(geo_gender_metric)

# %% [markdown]
# Now show the results per system:

# %%
geo_gender_rr = geo_gender_res.groupby('run_name').mean()
geo_gender_rr.sort_values('EE-L', inplace=True)
geo_gender_rr

# %% [markdown]
# ### Internal Properties
#
# This year, several of our properties are ‘internal’: that is, they primarily refer to things that matter within the Wikipedia platform, not broader social concerns.
#
# Let's see how the systems perform on those.

# %%
internal_names = ['alpha', 'age', 'pop', 'langs']
internal_dims = subset_dims(internal_names)
internal_tgt = subset_tgt(internal_names)

# %%
internal_metric = metrics.EELMetric(qrels.set_index('topic_id'), internal_dims, internal_tgt)

# %%
internal_res = runs.groupby(['run_name', 'topic_id']).progress_apply(internal_metric)

# %% [markdown]
# Now show the results per system:

# %%
internal_rr = internal_res.groupby('run_name').mean()
internal_rr.sort_values('EE-L', inplace=True)
internal_rr

# %% [markdown]
# ### Demographic Properties
#
# Let's see performance on the other ones (demographic properties):

# %%
demo_names = [d.name for d in dimensions if d.name not in internal_names]
demo_dims = subset_dims(demo_names)
demo_tgt = subset_tgt(demo_names)

# %%
demo_metric = metrics.EELMetric(qrels.set_index('topic_id'), demo_dims, demo_tgt)

# %%
demo_res = runs.groupby(['run_name', 'topic_id']).progress_apply(demo_metric)

# %% [markdown]
# Now show the results per system:

# %%
demo_rr = demo_res.groupby('run_name').mean()
demo_rr.sort_values('EE-L', inplace=True)
demo_rr

# %% [markdown]
# ### Subset Scores

# %%
subsets = {
    'Overall': run_scores,
    '2021': geo_gender_rr,
    'Internal': internal_rr,
    'Demographic': demo_rr,
}

# %%
ss_cols = [df[['EE-L']].rename(columns={'EE-L': name}) for (name, df) in subsets.items()]
ss_df = reduce(lambda df1, df2: df1.join(df2), ss_cols)
ss_df.sort_values('Overall', inplace=True)
ss_fmt = ss_df.style.highlight_min(props='font-weight: bold;').format(lambda x: '%.3f' % (x,))
ss_fmt

# %%
ss_fn = Path('figures/task2-subsets.tex')
style = ss_df.style.highlight_min(props='font: bold;').format(lambda x: '%.3f' % (x,))
style = style.format_index(axis=0, escape='latex')
style = style.hide(names=True)
ss_tex = style.to_latex()
ss_fn.write_text(ss_tex)
print(ss_tex)

# %%
