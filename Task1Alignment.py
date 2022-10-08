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
# # Task 1 Alignment
#
# This notebook computes the target distributions and retrieved page alignments for **Task 1**.
# It depends on the output of the PageAlignments notebook.

# %% [markdown]
# This notebook can be run in two modes: 'train', to process the training topics, and 'eval' for the eval topics.

# %% tags=["parameters"]
DATA_MODE = 'eval'

# %% [markdown]
# ## Setup
#
# We begin by loading necessary libraries:

# %%
import sys
import warnings
from collections import namedtuple
from functools import reduce
from itertools import product
import operator
from pathlib import Path

# %% tags=[]
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import json
from natural.size import binarysize
from natural.number import number

# %% [markdown]
# Set up progress bar and logging support:

# %%
from tqdm.auto import tqdm
tqdm.pandas(leave=False)

# %% tags=[]
import sys, logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger('Task1Alignment')

# %% [markdown]
# And set up an output directory:

# %%
from wptrec.save import OutRepo
output = OutRepo('data/metric-tables')

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
# ## Prep Overview
#
# Now that we have our alignments and qrels, we are ready to prepare the Task 1 metrics.
#
# We're first going to prepare the target distributions; then we will compute the alignments for the retrieved pages.

# %% [markdown]
# ## Subject Geography
#
# Subject geography targets the average of the relevant set alignments and the world population.

# %%
qr_sub_geo_align = qr_join(sub_geo_align)
qr_sub_geo_align

# %% [markdown]
# For purely geographic fairness, we just need to average the unknowns with the world pop:

# %%
qr_sub_geo_tgt = qr_sub_geo_align.groupby('topic_id').mean()
qr_sub_geo_fk = qr_sub_geo_tgt.iloc[:, 1:].sum('columns')
qr_sub_geo_tgt.iloc[:, 1:] *= 0.5
qr_sub_geo_tgt.iloc[:, 1:] += qr_sub_geo_fk.apply(lambda k: world_pop * k * 0.5)
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
# And repeat:

# %%
qr_src_geo_tgt = qr_src_geo_align.groupby('topic_id').mean()
qr_src_geo_fk = qr_src_geo_tgt.iloc[:, 1:].sum('columns')
qr_src_geo_tgt.iloc[:, 1:] *= 0.5
qr_src_geo_tgt.iloc[:, 1:] += qr_src_geo_fk.apply(lambda k: world_pop * k * 0.5)
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
qr_gender_tgt = qr_gender_align.groupby('topic_id').mean()
qr_gender_fk = qr_gender_tgt.iloc[:, 1:].sum('columns')
qr_gender_tgt.iloc[:, 1:] *= 0.5
qr_gender_tgt.iloc[:, 1:] += qr_gender_fk.apply(lambda k: gender_tgt * k * 0.5)
qr_gender_tgt.head()

# %%
output.save_table(qr_gender_tgt, f'task1-{DATA_MODE}-gender-target', parquet=True)

# %% [markdown]
# ## Remaining Attributes
#
# The remaining attributes don't need any further processing, as they aren't averaged.

# %%
qr_occ_align = qr_join(occ_align)
qr_occ_tgt = qr_occ_align.groupby('topic_id').sum()
qr_occ_tgt = norm_dist_df(qr_occ_tgt)
qr_occ_tgt.head()

# %%
output.save_table(qr_occ_tgt, f'task1-{DATA_MODE}-occ-target', parquet=True)

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
# ## Multidimensional Alignment
#
# Now, we need to set up the *multidimensional* alignment.  The basic version is just to multiply the targets, but that doesn't include the target averaging we want to do for geographic and gender targets.
#
# Doing that averaging further requires us to very carefully handle the unknown cases.
#
# We are going to proceed in three steps:
#
# 1. Define the averaged dimensions (with their background targets) and the un-averaged dimensions
# 2. Demonstrate the logic by working through the alignment computations for a single topic
# 3. Apply step (2) to all topics

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
# ### Demo
#
# To demonstrate how the logic works, let's first work it out in cells for one query (1).
#
# What are its documents?

# %% tags=[]
qno = qrels['topic_id'].iloc[0]
qdf = qrels[qrels['topic_id'] == qno]
qdf.name = qno
qdf

# %% [markdown]
# We can use these page IDs to get its alignments.

# %%
q_pages = qdf['page_id'].values

# %% [markdown]
# #### Accumulating Initial Targets

# %% [markdown]
# We're now going to grab the dimensions that have targets, and create a single xarray with all of them:

# %%
q_xta = reduce(operator.mul, [d.align.loc[q_pages] for d in avg_dims])
q_xta

# %% [markdown]
# We can similarly do this for the dimensions without targets:

# %%
q_raw_xta = reduce(operator.mul, [d.align.loc[q_pages] for d in raw_dims])
q_raw_xta

# %% [markdown]
# Now, we need to combine this with the other matrix to produce a complete alignment matrix, which we then will collapse into a query target matrix.  However, we don't have memory to do the whole thing at one go. Therefore, we will do it page by page.
#
# The `mean_outer` function does this:

# %%
from wptrec.dimension import mean_outer

# %%
q_tam = mean_outer(q_xta, q_raw_xta)
q_tam

# %%
q_tam

# %%
q_tam.sum()

# %% [markdown]
# In 2021, we ignored fully-unknown for Task 1. However, it isn't clear hot to properly do that with some attributes that are never fully unknown - they still need to be counted. Therefore, we consistently treat fully-unknown as a distinct category for both Task 1 and Task 2 metrics.

# %% [markdown]
# #### Data Subsetting
#
# Before we average, we need to be able to select data by its known/unknown status.
#
# Let's start by making a list of cases - the known/unknown status of each dimension.

# %%
avg_cases = list(product(*[[True, False] for d in avg_dims]))
avg_cases

# %% [markdown]
# The last entry is the all-unknown case - remove it:

# %%
avg_cases.pop()
avg_cases


# %% [markdown]
# We now want the ability to create an indexer to look up the subset of the alignment frame corresponding to a case. Let's write that function:

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
# Let's test this function quick:

# %%
case_selector(avg_cases[0])

# %%
case_selector(avg_cases[-1])

# %% [markdown]
# And make sure we can use it:

# %%
q_tam[case_selector(avg_cases[1])]

# %% [markdown]
# Fantastic! Given a case (known and unknown statuses), we can select the subset of the target matrix with exactly those.

# %% [markdown]
# #### Averaging
#
# Ok, now we have to - very carefully - average with our target modifier.  For each dimension that is not fully-unknown, we average with the intersectional target defined over the known dimensions.
#
# At all times, we also need to respect the fraction of the total it represents.
#
# We'll use the selection capabilities above to handle this.
#
# First, let's make sure that our target matrix sums to 1 to start with:

# %%
q_tam.sum()

# %% [markdown]
# Fantastic.  This means that if we sum up a subset of the data, it will give us the fraction of the distribution that has that combination of known/unknown status.
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
# And apply it:

# %%
q_target = avg_with_bg(q_tam, True)
q_target.sum()

# %%
q_target

# %%
print(number(q_target.values.size), 'values taking', binarysize(q_target.nbytes))

# %% [markdown]
# Is it still a distribution?

# %%
q_target.sum()

# %% [markdown]
# We can unravel this value into a single-dimensional array representing the multidimensional target:

# %%
q_target.values.ravel()


# %% [markdown]
# Now we have all the pieces to compute this for each of our queries.

# %% [markdown]
# ### Implementing Function
#
# To perform this combination for every query, we'll use a function that takes a data frame for a query's relevant docs and performs all of the above operations:

# %%
def query_xalign(pages):
    # compute targets to average
    avg_pages = reduce(operator.mul, [d.align.loc[pages] for d in avg_dims])
    raw_pages = reduce(operator.mul, [d.align.loc[pages] for d in raw_dims])

    # convert to query distribution
    tgt = mean_outer(avg_pages, raw_pages)

    # average with background distributions
    tgt = avg_with_bg(tgt)
    
    # and return the result
    return tgt

# %% [markdown]
# Make sure it works:

# %%
query_xalign(qdf.page_id.values)

# %% [markdown]
# ### Computing Query Targets

# %% [markdown]
# Now with that function, we can compute the alignment vector for each query.  Extract queries into a dictionary:

# %%
queries = {
    t: df['page_id'].values
    for (t, df) in qrels.groupby('topic_id')
}

# %% [markdown]
# Make an index that we'll need later for setting up the XArray dimension:

# %%
q_ids = pd.Index(queries.keys(), name='topic_id')
q_ids

# %% [markdown]
# Now let's create targets for each of these:

# %%
q_tgts = [query_xalign(queries[q]) for q in tqdm(q_ids)]

# %% [markdown]
# Assemble a composite xarray:

# %%
q_tgts = xr.concat(q_tgts, q_ids)
q_tgts

# %% [markdown]
# Save this to NetCDF (xarray's recommended format):

# %%
output.save_xarray(q_tgts, f'task1-{DATA_MODE}-int-targets')

# %%
