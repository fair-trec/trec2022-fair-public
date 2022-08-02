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
# # Task 2 Alignment
#
# This notebook computes the target distributions and retrieved page alignments for **Task 2**.
# It depends on the output of the PageAlignments notebook.

# %% [markdown]
# This notebook can be run in two modes: 'train', to process the training topics, and 'eval' for the eval topics.

# %% tags=["parameters"]
DATA_MODE = 'train'

# %% [markdown]
# ## Setup
#
# We begin by loading necessary libraries:

# %% tags=[]
import sys
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import json
from natural.size import binarysize

# %% [markdown]
# Set up progress bar and logging support:

# %%
from tqdm.auto import tqdm
tqdm.pandas(leave=False)

# %% tags=[]
import sys, logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger('Task2Alignment')

# %% [markdown]
# And set up an output directory:

# %%
from wptrec.save import OutRepo
output = OutRepo('data/metric-tables')

# %%
from wptrec import metrics

# %% [markdown]
# ## Data and Helpers
#
# Most data loading is outsourced to `MetricInputs`.  First we save the data mode where metric inputs can find it:

# %%
import wptrec
wptrec.DATA_MODE = DATA_MODE

# %%
from MetricInputs import *

# %%
dimensions


# %% [markdown]
# ### qrel join
#
# We want a function to join alignments with qrels:

# %%
def qr_join(align):
    return qrels.join(align, on='page_id').set_index(['topic_id', 'page_id'])


# %% [markdown]
# ### norm_dist
#
# And a function to normalize to a distribution:

# %%
def norm_dist_df(mat):
    sums = mat.sum('columns')
    return mat.divide(sums, 'rows')


# %% [markdown]
# ## Work and Target Exposure
#
# The first thing we need to do to prepare the metric is to compute the work-needed for each topic's pages, and use that to compute the target exposure for each (relevant) page in the topic.
#
# This is because an ideal ranking orders relevant documents in decreasing order of work needed, followed by irrelevant documents.  All relevant documents at a given work level should receive the same expected exposure.
#
# First, look up the work for each query page ('query page work', or qpw):

# %%
qpw = qrels.join(page_quality, on='page_id')
qpw

# %% [markdown]
# And now  use that to compute the number of documents at each work level:

# %%
qwork = qpw.groupby(['topic_id', 'quality'])['page_id'].count()
qwork


# %% [markdown]
# Now we need to convert this into target exposure levels.  This function will, given a series of counts for each work level, compute the expected exposure a page at that work level should receive.

# %%
def qw_tgt_exposure(qw_counts: pd.Series) -> pd.Series:
    if 'topic_id' == qw_counts.index.names[0]:
        qw_counts = qw_counts.reset_index(level='topic_id', drop=True)
    qwc = qw_counts.reindex(work_order, fill_value=0).astype('i4')
    tot = int(qwc.sum())
    da = metrics.discount(tot)
    qwp = qwc.shift(1, fill_value=0)
    qwc_s = qwc.cumsum()
    qwp_s = qwp.cumsum()
    res = pd.Series(
        [np.mean(da[s:e]) for (s, e) in zip(qwp_s, qwc_s)],
        index=qwc.index
    )
    return res


# %% [markdown]
# We'll then apply this to each topic, to determine the per-topic target exposures:

# %%
qw_pp_target = qwork.groupby('topic_id').apply(qw_tgt_exposure)
qw_pp_target.name = 'tgt_exposure'
qw_pp_target

# %% [markdown]
# We can now merge the relevant document work categories with this exposure, to compute the target exposure for each relevant document:

# %%
qp_exp = qpw.join(qw_pp_target, on=['topic_id', 'quality'])
qp_exp = qp_exp.set_index(['topic_id', 'page_id'])['tgt_exposure']
qp_exp

# %% [markdown]
# ## Subject Geography
#
# Subject geography targets the average of the relevant set alignments and the world population.

# %%
qr_sub_geo_align = qr_join(sub_geo_align)
qr_sub_geo_align

# %% [markdown]
# We can just average with the world pop, with a bit of normalization.

# %%
qr_sub_geo_tgt = qr_sub_geo_align.groupby('topic_id').sum()
qr_sub_geo_tgt = qr_sub_geo_tgt.iloc[:, 1:]
qr_sub_geo_tgt = norm_dist_df(qr_sub_geo_tgt)
qr_sub_geo_tgt = (qr_sub_geo_tgt + world_pop) * 0.5
qr_sub_geo_tgt.head()

# %% [markdown]
# Make sure the rows are distributions:

# %%
qr_sub_geo_tgt.sum('columns').describe()

# %% [markdown]
# Everything is 1, we're good to go!

# %%
output.save_table(qr_sub_geo_tgt, f'task1-{DATA_MODE}-sub-geo-target', parquet=True)

# %% [markdown]
# ## Source Geography
#
# Source geography works the same way.

# %%
qr_src_geo_align = qr_join(src_geo_align)
qr_src_geo_align

# %% [markdown]
# For purely geographic fairness, the target is easy - average with world pop.

# %%
qr_src_geo_tgt = qr_src_geo_align.groupby('topic_id').sum()
qr_src_geo_tgt = qr_src_geo_tgt.iloc[:, 1:]
qr_src_geo_tgt = norm_dist_df(qr_src_geo_tgt)
qr_src_geo_tgt = (qr_src_geo_tgt + world_pop) * 0.5
qr_src_geo_tgt.head()

# %% [markdown]
# Make sure the rows are distributions:

# %%
qr_src_geo_tgt.sum('columns').describe()

# %% [markdown]
# Everything is 1, we're good to go!

# %%
output.save_table(qr_src_geo_tgt, f'task1-{DATA_MODE}-src-geo-target', parquet=True)

# %% [markdown]
# ## Gender
#
# Now we're going to grab the gender alignments.  Again, we ignore UNKNOWN.

# %%
qr_gender_align = qr_join(gender_align)
qr_gender_align.head()

# %%
qr_gender_tgt = qr_gender_align.iloc[:, 1:].groupby('topic_id').sum()
qr_gender_tgt = norm_dist_df(qr_gender_tgt)
qr_gender_tgt = (qr_gender_tgt + gender_tgt) * 0.5
qr_gender_tgt.head()

# %%
output.save_table(qr_gender_tgt, f'task1-{DATA_MODE}-gender-target', parquet=True)

# %% [markdown]
# ## Occupation
#
# Occupation is more straightforward, since we don't have a global target to average with.  We do need to drop unknown.

# %%
qr_occ_align = qr_join(occ_align)
qr_occ_tgt = qr_occ_align.iloc[:, 1:].groupby('topic_id').sum()
qr_occ_tgt = norm_dist_df(qr_occ_tgt)
qr_occ_tgt.head()

# %%
output.save_table(qr_occ_tgt, f'task1-{DATA_MODE}-occ-target', parquet=True)

# %% [markdown]
# ## Remaining Attributes
#
# The remaining attributes don't need any further processing, as they are completely known.

# %%
qr_age_align = qr_join(age_align)
qr_age_tgt = norm_dist_df(qr_age_align.groupby('topic_id').sum())
output.save_table(qr_age_tgt, f'task1-{DATA_MODE}-age-target', parquet=True)

# %%
qr_alpha_align = qr_join(alpha_align)
qr_alpha_tgt = norm_dist_df(qr_alpha_align.groupby('topic_id').sum())
output.save_table(qr_alpha_tgt, f'task1-{DATA_MODE}-alpha-target', parquet=True)

# %%
qr_langs_align = qr_join(langs_align)
qr_langs_tgt = norm_dist_df(qr_langs_align.groupby('topic_id').sum())
output.save_table(qr_langs_tgt, f'task1-{DATA_MODE}-langs-target', parquet=True)

# %%
qr_pop_align = qr_join(pop_align)
qr_pop_tgt = norm_dist_df(qr_pop_align.groupby('topic_id').sum())
output.save_table(qr_pop_tgt, f'task1-{DATA_MODE}-pop-target', parquet=True)

# %% [markdown]
# ### Geographic Alignment
#
# Now that we've computed per-page target exposure, we're ready to set up the geographic alignment vectors for computing the per-*group* expected exposure with geographic data.
#
# We're going to start by getting the alignments for relevant documents for each topic:

# %%
qp_geo_align = qrels.join(page_geo_align, on='page_id').set_index(['id', 'page_id'])
qp_geo_align.index.names = ['q_id', 'page_id']
qp_geo_align

# %% [markdown]
# Now we need to compute the per-query target exposures.  This starst with aligning our vectors:

# %%
qp_geo_exp, qp_geo_align = qp_exp.align(qp_geo_align, fill_value=0)

# %% [markdown]
# And now we can multiply the exposure vector by the alignment vector, and summing by topic - this is equivalent to the matrix-vector multiplication on a topic-by-topic basis.

# %%
qp_aexp = qp_geo_align.multiply(qp_geo_exp, axis=0)
q_geo_align = qp_aexp.groupby('q_id').sum()

# %% [markdown]
# Now things get a *little* weird.  We want to average the empirical distribution with the world population to compute our fairness target.  However, we don't have empirical data on the distribution of articles that do or do not have geographic alignments.
#
# Therefore, we are going to average only the *known-geography* vector with the world population.  This proceeds in N steps:
#
# 1. Normalize the known-geography matrix so its rows sum to 1.
# 2. Average each row with the world population.
# 3. De-normalize the known-geography matrix so it is in the original scale, but adjusted w/ world population
# 4. Normalize the *entire* matrix so its rows sum to 1
#
# Let's go.

# %%
qg_known = q_geo_align.drop(columns=['Unknown'])

# %% [markdown]
# Normalize (adding a small value to avoid division by zero - affected entries will have a zero numerator anyway):

# %%
qg_ksums = qg_known.sum(axis=1)
qg_kd = qg_known.divide(np.maximum(qg_ksums, 1.0e-6), axis=0)

# %% [markdown]
# Average:

# %%
qg_kd = (qg_kd + world_pop) * 0.5

# %% [markdown]
# De-normalize:

# %%
qg_known = qg_kd.multiply(qg_ksums, axis=0)

# %% [markdown]
# Recombine with the Unknown column:

# %%
q_geo_tgt = q_geo_align[['Unknown']].join(qg_known)

# %% [markdown]
# Normalize targets:

# %%
q_geo_tgt = q_geo_tgt.divide(q_geo_tgt.sum(axis=1), axis=0)
q_geo_tgt

# %% [markdown]
# This is our group exposure target distributions for each query, for the geographic data.  We're now ready to set up the matrix.

# %%
train_geo_qtgt = q_geo_tgt.loc[train_topics['id']]
eval_geo_qtgt = q_geo_tgt.loc[eval_topics['id']]

# %% [markdown]
# And save data.

# %%
save_table(train_geo_qtgt, 'task2-train-geo-targets')
save_table(eval_geo_qtgt, 'task2-eval-geo-targets')


# %% [markdown]
# ### Intersectional Alignment
#
# Now we need to compute the intersectional targets for Task 2.  We're going to take a slightly different approach here, based on the intersectional logic for Task 1, because we've come up with better ways to write the code, but the effect is the same: only known aspects are averaged.
#
# We'll write a function very similar to the one for Task 1:

# %%
def query_xideal(qdf, ravel=True):
    pages = qdf['page_id']
    pages = pages[pages.isin(page_xalign.indexes['page'])]
    q_xa = page_xalign.loc[pages.values, :, :]
    
    # now we need to get the exposure for the pages, and multiply
    p_exp = qp_exp.loc[qdf.name]
    assert p_exp.index.is_unique
    p_exp = xr.DataArray(p_exp, dims=['page'])
    
    # and we multiply!
    q_xa = q_xa * p_exp

    # normalize into a matrix (this time we don't clear)
    q_am = q_xa.sum(axis=0)
    q_am = q_am / q_am.sum()
    
    # compute fractions in each section - combined with q_am[0,0], this should be about 1
    q_fk_all = q_am[1:, 1:].sum()
    q_fk_geo = q_am[1:, :1].sum()
    q_fk_gen = q_am[:1, 1:].sum()
    
    # known average
    q_am[1:, 1:] *= 0.5
    q_am[1:, 1:] += int_tgt * 0.5 * q_fk_all
    
    # known-geo average
    q_am[1:, :1] *= 0.5
    q_am[1:, :1] += geo_tgt_xa * 0.5 * q_fk_geo
    
    # known-gender average
    q_am[:1, 1:] *= 0.5
    q_am[:1, 1:] += gender_tgt_xa * 0.5 * q_fk_gen
    
    # and return the result
    if ravel:
        return pd.Series(q_am.values.ravel())
    else:
        return q_am


# %% [markdown]
# Test this function out:

# %%
query_xideal(qdf, ravel=False)

# %% [markdown]
# And let's go!

# %%
q_xtgt = qrels.groupby('id').progress_apply(query_xideal)
q_xtgt

# %%
train_qtgt = q_xtgt.loc[train_topics['id']]
eval_qtgt = q_xtgt.loc[eval_topics['id']]

# %% [markdown]
# And save our tables:

# %%
save_table(train_qtgt, 'task2-train-int-targets')

# %%
save_table(eval_qtgt, 'task2-eval-int-targets')

# %% [markdown]
# ## Task 2B - Equity of Underexposure
#
# For 2022, we are using a diffrent version of the metric. **Equity of Underexposure** looks at each page's underexposure (system exposure is less than target exposure), and looks for underexposure to be equitably distributed between groups.
#
# On its own, this isn't too difficult; averaging with background distributions, however, gets rather subtle.  Background distributions are at the roup level, but we need to propgagate that into the page level, so we can compute the difference between system and target exposure at the page level, and then aggregate the underexposure within each group.
#
# The idea of equity of underexposure is that we $\epsilon = \operatorname{E}_\pi [\eta]$ and $\epsilon^* = \operatorname{E}_\tau [\eta]$.  We then compute $u = min(\epsilon^* - \epsilon, 0)$, and restrict it to be negative, and aggregate it by group; if $A$ is our page alignment matrix and $\vec{u}$, we compute the group underexposure by $A^T \vec{u}$.
#
# That's the key idea.  However, we want to use $\epsilon^\dagger$ that has the equivalent of averaging group-aggregated $\epsilon^*$ with global target distributions $w_g$.  We can do this in a few stages.  First, we compute the total attention of each group, and use that to compute the fraction of group global weight that should go to each unit of alignment:
#
# \begin{align*}
# s_g & = \sum_d a_{dg} \\
# \hat{w}_g & = \frac{w_g}{s_g}
# \end{align*}
#
# We can then average:
#
# \begin{align*}
# \epsilon^\dagger_d & = \frac{1}{2}\left(\epsilon^*_d + \sum_g a_{dg} \hat{w}_g \epsilon^*_{\mathrm{total}} \right) \\
# \end{align*}
#
# This is all on a per-topic basis.

# %% [markdown]
# ### Demo Topic
#
# We're going to reuse demo topic data from before:

# %%
q_xa

# %% [markdown]
# Compute the total for each attribute:

# %%
s_xg = q_xa.sum(axis=0) + 1e-10
s_xg

# %% [markdown]
# Let's get some fractions out of that:

# %%
s_xgf = s_xg / s_xg.sum()
s_xgf

# %% [markdown]
# Now, let's make a copy, and start building up a world target matrix that properly accounts for missing values:

# %%
W = s_xgf.copy()

# %% [markdown]
# Now, let's put in the known intersectional targets:

# %%
W[1:, 1:] = int_tgt * W[1:, 1:].sum()

# %% [markdown]
# Now we need the known-gender / unknown-geo targets:

# %%
W[0, 1:] = int_tgt.sum(axis=0) * W[0, 1:].sum()

# %% [markdown]
# And the known-geo / unknown-gender targets:

# %%
W[1:, 0] = int_tgt.sum(axis=1) * W[1:, 0].sum()

# %% [markdown]
# Let's see what we have:

# %%
W

# %% [markdown]
# Now we normalize it by $s_g$:

# %%
Wh = W / s_xg
Wh

# %% [markdown]
# The massive values are only where we have no relevant items, so they'll never actually be used.
#
# We can now compute the query-aligned target matrix.

# %%
qp_gt = (q_xa * (Wh * qp_exp[1].sum())).sum(axis=(1,2)).to_series()
qp_gt.index.name = 'page_id'
qp_gt

# %%
qp_exp[1]

# %%
qp_tgt = 0.5 * (qp_exp[1] + qp_gt)
qp_tgt


# %% [markdown]
# ### Setting Up Matrix
#
# Now that we have the math worked out, we can create actual global target frames for each query.

# %%
def topic_page_tgt(qdf):
    pages = qdf['page_id']
    pages = pages[pages.isin(page_xalign.indexes['page'])]
    q_xa = page_xalign.loc[pages.values, :, :]
    
    # now we need to get the exposure for the pages
    p_exp = qp_exp.loc[qdf.name]
    assert p_exp.index.is_unique
    
    # need our sums
    s_xg = q_xa.sum(axis=0) + 1e-10
    
    # set up the global target
    W = s_xg / s_xg.sum()
    W[1:, 1:] = int_tgt * W[1:, 1:].sum()
    W[0, 1:] = int_tgt.sum(axis=0) * W[0, 1:].sum()
    W[1:, 0] = int_tgt.sum(axis=1) * W[1:, 0].sum()
    
    # per-unit global weights, de-normalized by total exposure
    Wh = W / s_xg
    Wh *= p_exp.sum()
    
    # compute global target
    gtgt = q_xa * Wh
    gtgt = gtgt.sum(axis=(1,2)).to_series()
    
    # compute average target
    avg_tgt = 0.5 * (p_exp + gtgt)
    avg_tgt.index.name = 'page'
    
    return avg_tgt


# %% [markdown]
# Test it quick:

# %%
topic_page_tgt(qdf)

# %% [markdown]
# And create our targets:

# %%
qp_tgt = qrels.groupby('id').progress_apply(topic_page_tgt)
qp_tgt

# %%
save_table(qp_tgt.to_frame('target'), 'task2-all-page-targets')

# %%
train_qptgt = qp_tgt.loc[train_topics['id']].to_frame('target')
eval_qptgt = qp_tgt.loc[eval_topics['id']].to_frame('target')

# %%
save_table(train_qptgt, 'task2-train-page-targets')
save_table(eval_qptgt, 'task2-eval-page-targets')
