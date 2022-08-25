# -*- coding: utf-8 -*-
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
# # Metric Inputs
#
# This notebook serves to *re-import* the metric input data (qrels, page alignments) that were prepared in `PageAlignments`.
# It can be imported as a Python module, and is intended to support the following usage in task-specific alignment & target
# notebooks:
#
# ```python
# from MetricImputs import *
# ```
#
# On its own, it just shows summaries of that data.

# %% [markdown]
# ## Setup
#
# Import some libraries:

# %%
import warnings
import logging
import pandas as pd
import xarray as xr
from pathlib import Path

# %% [markdown]
# We're now going to set up the data mode, if necessary.

# %%
import wptrec
DATA_MODE = getattr(wptrec, 'DATA_MODE', None)
if DATA_MODE is None:
    warnings.warn('No DATA_MODE specified, assuming ‘train’')
    DATA_MODE = 'train'

# %% [markdown]
# And the data dir

# %%
DATA_DIR = Path('data/metric-tables')

# %%
_log = logging.getLogger(__name__)

# %% [markdown]
# ## Topics
#
# Now we will load the topics:

# %%
topics = pd.read_json(f'data/trec_2022_{DATA_MODE}_reldocs.jsonl', lines=True)
topics.head()

# %%
topics.rename(columns={'id': 'topic_id'}, inplace=True)

# %% [markdown]
# Now we are going to explode this into a set of `qrels`:

# %%
qrels = topics[['topic_id', 'rel_docs']].explode('rel_docs', ignore_index=True)
qrels.rename(columns={'rel_docs': 'page_id'}, inplace=True)
qrels['page_id'] = qrels['page_id'].astype('i4')
qrels = qrels.drop_duplicates()
qrels.head()


# %% [markdown]
# ## Page Alignments
#
# And the page alignments, with a helper function.

# %%
def _load_page_align(key):
    fn = DATA_DIR / f'page-{key}-align.parquet'
    _log.info('reading %s', fn)
    df = pd.read_parquet(fn)
    df.index.name = 'page_id'
    df.name = key
    dfx = xr.DataArray(df, dims=['page', key])
    return df, dfx


# %%
sub_geo_align, sub_geo_xr = _load_page_align('sub-geo')

# %%
src_geo_align, src_geo_xr = _load_page_align('src-geo')

# %%
gender_align, gender_xr = _load_page_align('gender')

# %%
occ_align, occ_xr = _load_page_align('occ')

# %%
alpha_align, alpha_xr = _load_page_align('alpha')

# %%
age_align, age_xr = _load_page_align('age')

# %%
pop_align, pop_xr = _load_page_align('pop')

# %%
langs_align, langs_xr = _load_page_align('langs')

# %% [markdown]
# ## Geographic Background
#
# Our geographic target needs world population for to establish an equity target - this data comes from Wikipedia's [List of continents and continental subregions by population](https://en.wikipedia.org/wiki/List_of_continents_and_continental_subregions_by_population).

# %%
world_pop = pd.read_csv('data/world-pop.csv')
world_pop

# %% [markdown]
# Process it into a distribution series:

# %%
world_pop = world_pop.set_index('Name')['Population']
world_pop /= world_pop.sum()
world_pop.name = 'geography'
world_pop.sort_index(inplace=True)
world_pop

# %% [markdown]
# ## Gender Background
#
# And a gender global target:

# %%
gender_tgt = pd.Series({
    'female': 0.495,
    'male': 0.495,
    'NB': 0.01
})
gender_tgt.name = 'gender'
gender_tgt.sum()

# %% [markdown]
# ## Static Data
#
# The work-needed codes have an order:

# %%
work_order = [
    'Stub',
    'Start',
    'C',
    'B',
    'GA',
    'FA',
]

# %% [markdown]
# And finally a name for unknown:

# %%
UNKNOWN = '@UNKNOWN'

# %% [markdown]
# ## Page Quality
#
# And we can load the page quality data:

# %%
page_quality = pd.read_parquet(DATA_DIR / 'page-quality.parquet')
page_quality = page_quality.set_index('page_id')['quality']
page_quality = page_quality.astype('category').cat.reorder_categories(work_order)
page_quality = page_quality.to_frame()

# %% [markdown]
# ## Dimension Lists
#
# We're going to make a list of dimensions, along with their targets.
# We have a class to define these:

# %%
from wptrec.dimension import FairDim

# %%
dimensions = [
    FairDim(sub_geo_align, sub_geo_xr, world_pop, True),
    FairDim(src_geo_align, src_geo_xr, world_pop, True),
    FairDim(gender_align, gender_xr, gender_tgt, True),
    FairDim(occ_align, occ_xr, None, True),
    FairDim(alpha_align, alpha_xr),
    FairDim(age_align, age_xr),
    FairDim(pop_align, pop_xr),
    FairDim(langs_align, langs_xr),
]

# %%
dimensions
