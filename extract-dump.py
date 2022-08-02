"""
Extract a subset in JSON from a dump file.

Usage:
    extract-dump.py [options] DUMP

Options:
    DUMP
        Process dump file DUMP
    -o FILE
        Write output to FILE [default: trec_corpus.json.gz]
    --verbose
        Turn on verbose logging.
"""

import logging
import sys
from contextlib import contextmanager
from os import fspath
from pathlib import Path
import subprocess as sp
import json
import bz2, gzip
from docopt import docopt

import mwxml
from tqdm import tqdm

_log = logging.getLogger('extract-dump')


def load_dump(dump_file: Path):
    with bz2.open(dump_file) as bzf:
        dump = mwxml.Dump.from_file(bzf)
        for page in dump:
            if page.namespace == 0 and not page.redirect:
                rev = next(page)
                text = rev.text
                yield page.id, page.title, text


def process_dump(dump_file: Path, json_file: Path):
    _log.info('saving to %s', json_file)

    with gzip.open(json_file, 'wb', 9) as jsf:
        _log.info('reading %s', dump_file)
        for id, title, text in tqdm(load_dump(dump_file)):
            ut = title.replace(' ', '_')
            obj = {'id': id, 'title': title, 'text': text, 'url': f'https://en.wikipedia.org/wiki/{ut}'}
            jsf.write(json.dumps(obj).encode('utf8'))
            jsf.write(b'\n')
    

def main(opts):
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

    file = Path(opts['DUMP'])
    out = Path(opts['-o'])
    process_dump(file, out)



if __name__ == '__main__':
    opts = docopt(__doc__)
    main(opts)
