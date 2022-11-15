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
# It depends on the output of the PageAlignments notebook, as imported by MetricInputs.

# %% [markdown]
# This notebook can be run in two modes: 'train', to process the training topics, and 'eval' for the eval topics.

# %% tags=["parameters"]
DATA_MODE = 'eval'

# %% [markdown]
# ## Setup
#
# We begin by loading necessary libraries:

# %% tags=[]
import sys
import operator
from functools import reduce
from itertools import product
from collections import namedtuple
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
from wptrec.dimension import sum_outer

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
# Compute a raw target, factoring in weights:

# %%
qr_sub_geo_tgt = qr_sub_geo_align.multiply(qp_exp, axis='rows').groupby('topic_id').sum()

# %% [markdown]
# And now we need to average the known-geo with the background.

# %%
qr_sub_geo_fk = qr_sub_geo_tgt.iloc[:, 1:].sum('columns')
qr_sub_geo_tgt.iloc[:, 1:] *= 0.5
qr_sub_geo_tgt.iloc[:, 1:] += qr_sub_geo_fk.apply(lambda k: world_pop * k * 0.5)
qr_sub_geo_tgt.head()

# %% [markdown]
# These are **not** distributions, let's fix that!

# %%
qr_sub_geo_tgt = norm_dist_df(qr_sub_geo_tgt)

# %%
output.save_table(qr_sub_geo_tgt, f'task2-{DATA_MODE}-sub-geo-target', parquet=True)

# %% [markdown]
# ## Source Geography
#
# Source geography works the same way.

# %%
qr_src_geo_align = qr_join(src_geo_align)
qr_src_geo_align

# %% [markdown]
# And now we repeat these computations!

# %%
qr_src_geo_tgt = qr_src_geo_align.multiply(qp_exp, axis='rows').groupby('topic_id').sum()

# %%
qr_src_geo_fk = qr_src_geo_tgt.iloc[:, 1:].sum('columns')
qr_src_geo_tgt.iloc[:, 1:] *= 0.5
qr_src_geo_tgt.iloc[:, 1:] += qr_src_geo_fk.apply(lambda k: world_pop * k * 0.5)
qr_src_geo_tgt.head()

# %% [markdown]
# Make sure the rows are distributions:

# %%
qr_src_geo_tgt = norm_dist_df(qr_src_geo_tgt)

# %%
output.save_table(qr_src_geo_tgt, f'task2-{DATA_MODE}-src-geo-target', parquet=True)

# %% [markdown]
# ## Gender
#
# Now we're going to grab the gender alignments.  Works the same way.

# %%
qr_gender_align = qr_join(gender_align)
qr_gender_align.head()

# %%
qr_gender_tgt = qr_gender_align.multiply(qp_exp, axis='rows').groupby('topic_id').sum()

# %%
qr_gender_fk = qr_gender_tgt.iloc[:, 1:].sum('columns')
qr_gender_tgt.iloc[:, 1:] *= 0.5
qr_gender_tgt.iloc[:, 1:] += qr_gender_fk.apply(lambda k: gender_tgt * k * 0.5)
qr_gender_tgt.head()

# %%
qr_gender_tgt = norm_dist_df(qr_gender_tgt)

# %%
output.save_table(qr_gender_tgt, f'task2-{DATA_MODE}-gender-target', parquet=True)

# %% [markdown]
# ## Occupation
#
# Occupation is more straightforward, since we don't have a global target to average with.

# %%
qr_occ_align = qr_join(occ_align).multiply(qp_exp, axis='rows')
qr_occ_tgt = qr_occ_align.groupby('topic_id').sum()
qr_occ_tgt = norm_dist_df(qr_occ_tgt)
qr_occ_tgt.head()

# %%
output.save_table(qr_occ_tgt, f'task2-{DATA_MODE}-occ-target', parquet=True)

# %% [markdown]
# ## Remaining Attributes
#
# The remaining attributes don't need any further processing, as they are completely known.

# %%
qr_age_align = qr_join(age_align).multiply(qp_exp, axis='rows')
qr_age_tgt = norm_dist_df(qr_age_align.groupby('topic_id').sum())
output.save_table(qr_age_tgt, f'task2-{DATA_MODE}-age-target', parquet=True)

# %%
qr_alpha_align = qr_join(alpha_align).multiply(qp_exp, axis='rows')
qr_alpha_tgt = norm_dist_df(qr_alpha_align.groupby('topic_id').sum())
output.save_table(qr_alpha_tgt, f'task2-{DATA_MODE}-alpha-target', parquet=True)

# %%
qr_langs_align = qr_join(langs_align).multiply(qp_exp, axis='rows')
qr_langs_tgt = norm_dist_df(qr_langs_align.groupby('topic_id').sum())
output.save_table(qr_langs_tgt, f'task2-{DATA_MODE}-langs-target', parquet=True)

# %%
qr_pop_align = qr_join(pop_align).multiply(qp_exp, axis='rows')
qr_pop_tgt = norm_dist_df(qr_pop_align.groupby('topic_id').sum())
output.save_table(qr_pop_tgt, f'task2-{DATA_MODE}-pop-target', parquet=True)

# %% [markdown]
# ## Multidimensional Alignment
#
# Now let's dive into the multidmensional alignment.  This is going to proceed a lot like the Task 1 alignment.

# %% [markdown]
# ### Dimension Definitions
#
# Let's define background distributions for some of our dimensions:

# %%
dim_backgrounds = {
    'sub-geo': world_pop,
    'src-geo': world_pop,
    'gender': gender_tgt,
}

# %% [markdown]
# Now we'll make a list of dimensions to treat with averaging:

# %%
DR = namedtuple('DimRec', ['name', 'align', 'background'], defaults=[None])
avg_dims = [
    DR(d.name, d.page_align_xr, xr.DataArray(dim_backgrounds[d.name], dims=[d.name]))
    for d in dimensions
    if d.name in dim_backgrounds
]
[d.name for d in avg_dims]

# %% [markdown]
# And a list of dimensions to use as-is:

# %%
raw_dims = [
    DR(d.name, d.page_align_xr)
    for d in dimensions
    if d.name not in dim_backgrounds
]
[d.name for d in raw_dims]

# %% [markdown]
# Now: these dimension are in the original order - `dimensions` has the averaged dimensions before the non-averaged ones. **This is critical for the rest of the code to work.**

# %% [markdown]
# ### Data Subsetting
#
# Also from Task 1.

# %%
avg_cases = list(product(*[[True, False] for d in avg_dims]))
avg_cases.pop()
avg_cases


# %%
def case_selector(case):
    def mksel(known):
        if known:
            # select all but 1st column
            return slice(1, None, None)
        else:
            # select 1st column
            return 0
    
    return tuple(mksel(k) for k in case)


# %% [markdown]
# ### Background Averaging
#
# We're now going to define our background-averaging function; this is reused from the Task 1 alignment code.
#
# For each condition, we are going to proceed as follows:
#
# 1. Compute an appropriate intersectional background distribution (based on the dimensions that are "known")
# 2. Select the subset of the target matrix with this known status
# 3. Compute the sum of this subset
# 4. Re-normalize the subset to sum to 1
# 5. Compute a normalization table such that each coordinate in the distributions to correct sums to 1 (so multiplying this by the background distribution spreads the background across the other dimensions appropriately), and use this to spread the background distribution
# 6. Average with the spread background distribution
# 7. Re-normalize to preserve the original sum
#
# Let's define the whole process as a function:

# %%
def avg_with_bg(tm, verbose=False):
    tm = tm.copy()
    
    tail_names = [d.name for d in raw_dims]
    
    # compute the tail mass for each coordinate (can be done once)
    tail_mass = tm.sum(tail_names)
    
    # now some things don't have any mass, but we still need to distribute background distributions.
    # solution: we impute the marginal tail distribution
    # first compute it
    tail_marg = tm.sum([d.name for d in avg_dims])
    # then impute that where we don't have mass
    tm_imputed = xr.where(tail_mass > 0, tm, tail_marg)
    # and re-compute the tail mass
    tail_mass = tm_imputed.sum(tail_names)
    # and finally we compute the rescaled matrix
    tail_scale = tm_imputed / tail_mass
    del tm_imputed
    
    for case in avg_cases:
        # for deugging: get names
        known_names = [d.name for (d, known) in zip(avg_dims, case) if known]
        if verbose:
            print('processing known:', known_names)
        
        # Step 1: background
        bg = reduce(operator.mul, [
            d.background
            for (d, known) in zip(avg_dims, case)
            if known
        ])
        if not np.allclose(bg.sum(), 1.0):
            warnings.warn('background distribution for {} sums to {}, expected 1'.format(known_names, bg.values.sum()))
        
        # Step 2: selector
        sel = case_selector(case)
        
        # Steps 3: sum in preparation for normalization
        c_sum = tm[sel].sum()
        
        # Step 5: spread the background
        bg_spread = bg * tail_scale[sel] * c_sum
        if not np.allclose(bg_spread.sum(), c_sum):
            warnings.warn('rescaled background sums to {}, expected c_sum'.format(bg_spread.values.sum()))
        
        # Step 4 & 6: average with the background
        tm[sel] *= 0.5
        bg_spread *= 0.5
        tm[sel] += bg_spread
                        
        if not np.allclose(tm[sel].sum(), c_sum):
            warnings.warn('target distribution for {} sums to {}, expected {}'.format(known_names, tm[sel].values.sum(), c_sum))
    
    return tm


# %% [markdown]
# ### Computing Targets
#
# We're now ready to compute a multidimensional target. This works like the Task 1, with the difference that we are propagating work needed into the targets as well; the input will be series whose *index* is page IDs and values are the work levels.

# %%
def query_xalign(pages):
    # compute targets to average
    avg_pages = reduce(operator.mul, [d.align.loc[pages.index] for d in avg_dims])
    raw_pages = reduce(operator.mul, [d.align.loc[pages.index] for d in raw_dims])
    
    # weight the left pages
    pages.index.name = 'page'
    qpw = xr.DataArray.from_series(pages)
    avg_pages = avg_pages * qpw

    # convert to query distribution
    tgt = sum_outer(avg_pages, raw_pages)
    tgt /= qpw.sum()

    # average with background distributions
    tgt = avg_with_bg(tgt)
    
    # and return the result
    return tgt


# %% [markdown]
# ### Applying Computations
#
# Now let's run this thing - compute all the target distributions:

# %%
q_ids = qp_exp.index.levels[0].copy()
q_ids

# %%
q_tgts = [query_xalign(qp_exp.loc[q]) for q in tqdm(q_ids)]

# %%
q_tgts = xr.concat(q_tgts, q_ids)
q_tgts

# %% [markdown]
# Save this to NetCDF (xarray's recommended format):

# %%
output.save_xarray(q_tgts, f'task2-{DATA_MODE}-int-targets')

# %%
