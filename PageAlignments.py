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
# # Page Alignments
#
# This notebook computes the *page alignments* from the Wikipedia metadata.  These are then used by the
# task-specific alignment notebooks to compute target distributions and page alignment subsets for retrieved pages.
#
# **Warning:** this notebook takes quite a bit of memory to run.

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
log = logging.getLogger('PageAlignments')

# %% [markdown]
# And set up an output directory:

# %%
from wptrec.save import OutRepo
output = OutRepo('data/metric-tables')

# %% [markdown]
# ## Loading Data
#
# Now we need to load the data.

# %% [markdown]
# ### Static Data
# We need a set of subregions that are folded into [Oceania](https://en.wikipedia.org/wiki/United_Nations_geoscheme_for_Oceania):

# %%
oc_regions = [
    'Australia and New Zealand',
    'Melanesia',
    'Micronesia',
    'Polynesia',
]

# %% [markdown]
# And finally a name for unknown:

# %%
UNKNOWN = '@UNKNOWN'

# %% [markdown]
# Now all our background data is set up.

# %% [markdown]
# ### Page Data
#
# Finally, we load the page metadata.  This is a little manual to manage memory usage.  Two memory usage tricks:
#
# - Only import the things we need
# - Use `sys.intern` for strings representing categoricals to decrease memory use
#
# Bonus is that, through careful logic, we get a progress bar.

# %%
page_path = Path('data/trec_2022_articles_discrete.json.gz')
page_file_size = page_path.stat().st_size
binarysize(page_file_size)

# %% [markdown]
# #### Definitions
#
# Let's define the different attributes we need to extract:

# %%
SUB_GEO_ATTR = 'page_subcont_regions'
SRC_GEO_ATTR = 'source_subcont_regions'
GENDER_ATTR = 'gender'
OCC_ATTR = 'occupations'
BASIC_ATTRS = [
    'page_id',
    'first_letter_category',
    'creation_date_category',
    'relative_pageviews_category',
    'num_sitelinks_category',
]

# %% [markdown]
# #### Read Data
#
# Now, we're going to process by creating lists we can reassemble with `pd.DataFrame.from_records`.  We'll fill these with tuples and dictionaries as appropriate.

# %%
qual_recs = []
sub_geo_recs = []
src_geo_recs = []
gender_recs = []
occ_recs = []
att_recs = []
seen_pages = set()

# %% [markdown]
# And we're off.

# %%
with tqdm(total=page_file_size, desc='compressed input', unit='B', unit_scale=True) as fpb:
    with open(page_path, 'rb') as gzf, gzip.GzipFile(fileobj=gzf, mode='r') as decoded:
        for line in decoded:
            line = json.loads(line)
            page = line['page_id']
            if page in seen_pages:
                continue
            else:
                seen_pages.add(page)
            
            # page quality
            qual_recs.append((page, line['qual_cat']))
            
            # page geography
            for geo in line[SUB_GEO_ATTR]:
                sub_geo_recs.append((page, sys.intern(geo)))
            
            # src geography
            psg = {'page_id': page}
            for g, v in line[SRC_GEO_ATTR].items():
                if g == 'UNK':
                    g = UNKNOWN
                psg[sys.intern(g)] = v
            src_geo_recs.append(psg)
            
            # genders
            for g in line[GENDER_ATTR]:
                gender_recs.append((page, sys.intern(g)))
            
            # occupations
            for occ in line[OCC_ATTR]:
                occ_recs.append((page, sys.intern(occ)))
            
            # other attributes
            att_recs.append(tuple((sys.intern(line[a]) if isinstance(line[a], str) else line[a])
                                  for a in BASIC_ATTRS))
            
            fpb.update(gzf.tell() - fpb.n)  # update the progress bar

# %% [markdown]
# #### Reassemble DFs
#
# Now we will assemble these records into data frames.

# %%
quality = pd.DataFrame.from_records(qual_recs, columns=['page_id', 'quality'])

# %%
sub_geo = pd.DataFrame.from_records(sub_geo_recs, columns=['page_id', 'sub_geo'])
sub_geo.info()

# %%
src_geo = pd.DataFrame.from_records(src_geo_recs)
src_geo.info()

# %%
gender = pd.DataFrame.from_records(gender_recs, columns=['page_id', 'gender'])
gender.info()

# %%
occupations = pd.DataFrame.from_records(occ_recs, columns=['page_id', 'occ'])
occupations.info()

# %%
cat_attrs = pd.DataFrame.from_records(att_recs, columns=BASIC_ATTRS)
cat_attrs.info()

# %%
all_pages = np.array(list(seen_pages))
all_pages = np.sort(all_pages)
all_pages = pd.Series(all_pages)

# %%
del src_geo_recs, sub_geo_recs
del gender_recs, occ_recs
del seen_pages

# %%
# %reset -f out

# %%
import gc
gc.collect()


# %% [markdown]
# ## Helper Functions
#
# These functions will help with further computations.

# %% [markdown]
# ### Normalize Distribution
#
# We are going to compute a number of data frames that are alignment vectors, such that each row is to be a multinomial distribution.  This function
# normalizes such a frame.

# %%
def norm_align_matrix(df):
    df = df.fillna(0)
    sums = df.sum(axis='columns')
    return df.div(sums, axis='rows')

# %% [markdown]
# ## Page Alignments
#
# All of our metrics require page "alignments": the protected-group membership of each page.

# %% [markdown]
# ### Quality
#
# Quality isn't an alignment, but we're going to save it here:

# %%
output.save_table(quality, 'page-quality', parquet=True)

# %% [markdown]
# ### Page Geography
#
# Let's start with the straight page geography alignment for the public evaluation of the training queries.  We've already loaded it above.
#
# We need to do a little cleanup on this data:
#
# - Align pages with no known geography with '@UNKNOWN' (to sort before known categories)
# - Replace Oceania subregions with Oceania

# %%
sub_geo.head()

# %% [markdown]
# Let's start by turning this into a wide frame:

# %%
sub_geo_align = sub_geo.assign(x=1).pivot(index='page_id', columns='sub_geo', values='x')
sub_geo_align.fillna(0, inplace=True)
sub_geo_align.head()

# %% [markdown]
# Now we need to collapse Oceania into one column.

# %%
ocean = sub_geo_align.loc[:, oc_regions].sum(axis='columns')
sub_geo_align = sub_geo_align.drop(columns=oc_regions)
sub_geo_align['Oceania'] = ocean

# %% [markdown]
# Next we need to add the Unknown column and expand this.
#
# Sum the items to find total amounts, and then create a series for unknown:

# %%
sub_geo_sums = sub_geo_align.sum(axis='columns')
sub_geo_unknown = ~(sub_geo_sums > 0)
sub_geo_unknown = sub_geo_unknown.astype('f8')
sub_geo_unknown = sub_geo_unknown.reindex(all_pages, fill_value=1)

# %% [markdown]
# Now let's join this with the original frame:

# %%
sub_geo_align = sub_geo_unknown.to_frame(UNKNOWN).join(sub_geo_align, how='left')
sub_geo_align = norm_align_matrix(sub_geo_align)
sub_geo_align.head()

# %%
sub_geo_align.sort_index(axis='columns', inplace=True)
sub_geo_align.info()

# %% [markdown]
# And convert this to an xarray for multidimensional usage:

# %% tags=[]
sub_geo_xr = xr.DataArray(sub_geo_align, dims=['page', 'sub_geo'])
sub_geo_xr

# %% tags=[]
binarysize(sub_geo_xr.nbytes)

# %%
output.save_table(sub_geo_align, 'page-sub-geo-align', parquet=True)

# %% [markdown]
# ### Page Source Geography
#
# We now need to do a similar setup for page source geography, which comes to us as a multinomial distribution already.

# %%
src_geo.head()

# %% [markdown]
# Set up the index:

# %%
src_geo.set_index('page_id', inplace=True)

# %% [markdown]
# Expand, then put 1 in UNKNOWN for everything that's missing:

# %%
src_geo_align = src_geo.reindex(all_pages, fill_value=0)
src_geo_align.loc[src_geo_align.sum('columns') == 0, UNKNOWN] = 1
src_geo_align

# %% [markdown]
# Collapse Oceania:

# %%
ocean = src_geo_align.loc[:, oc_regions].sum(axis='columns')
src_geo_align = src_geo_align.drop(columns=oc_regions)
src_geo_align['Oceania'] = ocean

# %% [markdown]
# And normalize.

# %%
src_geo_align = norm_align_matrix(src_geo_align)

# %%
src_geo_align.sort_index(axis='columns', inplace=True)
src_geo_align.info()

# %% [markdown]
# Xarray:

# %%
src_geo_xr = xr.DataArray(src_geo_align, dims=['page', 'src_geo'])
src_geo_xr

# %% [markdown]
# And save:

# %%
output.save_table(src_geo_align, 'page-src-geo-align', parquet=True)

# %% [markdown]
# ### Gender
#
# Now let's work on extracting gender - this is going work a lot like page geography.

# %% tags=[]
gender.head()

# %% [markdown]
# And summarize:

# %%
gender['gender'].value_counts()

# %% [markdown]
# Now, we're going to do a little more work to reduce the dimensionality of the space.  Points:
#
# 1. Trans men are men
# 2. Trans women are women
# 3. Cis/trans status is an adjective that can be dropped for the present purposes
#
# The result is that we will collapse "transgender female" and "cisgender female" into "female".
#
# The **downside** to this is that trans men are probabily significantly under-represented, but are now being collapsed into the dominant group.

# %% tags=[]
pgcol = gender['gender']
pgcol = pgcol.str.replace(r'(?:tran|ci)sgender\s+((?:fe)?male)', r'\1', regex=True)
pgcol.value_counts()

# %% [markdown]
# Now, we're going to group the remaining gender identities together under the label 'NB'.  As noted above, this is a debatable exercise that collapses a lot of identity.

# %% tags=[]
gender_labels = [UNKNOWN, 'female', 'male', 'NB']
pgcol[~pgcol.isin(gender_labels)] = 'NB'
pgcol.value_counts()

# %% [markdown]
# Now put this column back in the frame and deduplicate.

# %% tags=[]
page_gender = gender.assign(gender=pgcol)
page_gender = page_gender.drop_duplicates()

# %%
del pgcol

# %% [markdown]
# Now we need to add unknown genders.

# %%
kg_mask = all_pages.isin(page_gender['page_id'])
unknown = all_pages[~kg_mask]
page_gender = pd.concat([
    page_gender,
    pd.DataFrame({'page_id': unknown, 'gender': UNKNOWN})
], ignore_index=True)
page_gender

# %% [markdown]
# And make an alignment matrix:

# %% tags=[]
gender_align = page_gender.reset_index().assign(x=1).pivot(index='page_id', columns='gender', values='x')
gender_align.fillna(0, inplace=True)
gender_align = gender_align.reindex(columns=gender_labels)
gender_align.head()

# %% [markdown]
# Let's see how frequent each of the genders is:

# %% tags=[]
gender_align.sum(axis=0).sort_values(ascending=False)

# %% [markdown]
# And convert to an xarray:

# %% tags=[]
gender_xr = xr.DataArray(gender_align, dims=['page', 'gender'])
gender_xr

# %% tags=[]
binarysize(gender_xr.nbytes)

# %%
output.save_table(gender_align, 'page-gender-align', parquet=True)

# %% [markdown]
# ### Occupation
#
# Occupation works like gender, but without the need for processing.
#
# Convert to a matrix:

# %%
occ_align = occupations.assign(x=1).pivot(index='page_id', columns='occ', values='x')
occ_align.head()

# %% [markdown]
# Set up unknown and merge:

# %%
occ_unk = pd.Series(1.0, index=all_pages)
occ_unk.index.name = 'page_id'
occ_kmask = all_pages.isin(occ_align.index)
occ_kmask.index = all_pages
occ_unk[occ_kmask] = 0
occ_align = occ_unk.to_frame(UNKNOWN).join(occ_align, how='left')
occ_align = norm_align_matrix(occ_align)
occ_align.head()

# %%
occ_xr = xr.DataArray(occ_align, dims=['page', 'occ'])
occ_xr

# %% [markdown]
# And save:

# %%
output.save_table(occ_align, 'page-occ-align', parquet=True)

# %% [markdown]
# ### Other Attributes
#
# The other attributes don't require as much re-processing - they can be used as-is as categorical variables.  Let's save!

# %%
pages = cat_attrs.set_index('page_id')
pages

# %% [markdown]
# Now each of these needs to become another table.  The `get_dummies` function is our friend.

# %%
alpha_align = pd.get_dummies(pages['first_letter_category'])

# %%
output.save_table(alpha_align, 'page-alpha-align', parquet=True)

# %%
alpha_xr = xr.DataArray(alpha_align, dims=['page', 'alpha'])

# %%
age_align = pd.get_dummies(pages['creation_date_category'])
output.save_table(age_align, 'page-age-align', parquet=True)

# %%
age_xr = xr.DataArray(age_align, dims=['page', 'age'])

# %%
pop_align = pd.get_dummies(pages['relative_pageviews_category'])
output.save_table(pop_align, 'page-pop-align', parquet=True)

# %%
pop_xr = xr.DataArray(pop_align, dims=['page', 'pop'])

# %%
langs_align = pd.get_dummies(pages['num_sitelinks_category'])
output.save_table(langs_align, 'page-langs-align', parquet=True)

# %%
langs_xr = xr.DataArray(langs_align, dims=['page', 'langs'])

# %% [markdown]
# ## Working with Alignments
#
# At this point, we have computed an alignment matrix for each of our attributes, and extracted the qrels.
#
# We will use the data saved from this in separate notebooks to compute targets and alignments for tasks.
