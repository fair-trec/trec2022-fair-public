"""
Utilities for saving data (esp. alignment data).
"""

import logging
from pathlib import Path
from natural.size import binarysize

_log = logging.getLogger(__name__)


def log_file(fn):
    stat = fn.stat()
    _log.info('%s: %s', fn, binarysize(stat.st_size))


class OutRepo:
    """
    Class for storing outputs.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True, parents=True)
    
    def save_table(self, tbl, name, *, parquet=False, **kwargs):
        csv_fn = self.path / f'{name}.csv.gz'
        _log.info('saving CSV to %s', csv_fn)
        tbl.to_csv(csv_fn, **kwargs)
        log_file(csv_fn)
        
        if parquet:
            pq_fn = self.path / f'{name}.parquet'
            _log.info('saving Parquet to %s', pq_fn)
            tbl.to_parquet(pq_fn, compression='zstd', **kwargs)
            log_file(pq_fn)

    def save_xarray(self, array, name):
        nc_file = self.path / f'{name}.nc'
        _log.info('saving NetCDF to %s', nc_file)
        array.to_netcdf(nc_file)