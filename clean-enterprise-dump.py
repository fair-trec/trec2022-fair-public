"""
Clean up a full Wikipedia enterprise dump for 2022 and later.

Usage:
    clean-enterprise-dump.py [options] DUMP

Options:
    DUMP
        Process dump file DUMP
    -j THREADS
        Run with THREADS workers
    --verbose
        Turn on verbose logging
"""

import logging
import sys
from contextlib import contextmanager
from os import fspath
from pathlib import Path
import json
import gzip
import tarfile
from this import d
from docopt import docopt
from threading import Thread
from multiprocessing import Process, Queue, cpu_count
from subprocess import Popen, PIPE

from tqdm import tqdm

from wptrec.clean import parse_and_clean_wikitext

_log = logging.getLogger('clean-enterprise-dump')

class Sink:
    def __init__(self, file, key) -> None:
        self.key = key
        self.keep = ['id', 'title', 'url'] + [key]
        outf = open(file, 'wb')
        self.process = Popen(['gzip', '-9'], stdout=outf, stdin=PIPE)
        outf.close()
    
    def write(self, bytes):
        return self.process.stdin.write(bytes)

    def put(self, obj):
        obj = { k: v for (k, v) in obj.items() if k in self.keep }
        self.write(json.dumps(obj).encode('utf8'))
        self.write(b'\n')
    
    def close(self):
        self.process.stdin.close()
        self.process.wait()
        if self.process.returncode:
            print('gzip exited with code', self.process.returncode, file=sys.stderr)


class ReturnThread:
    def __init__(self, queue, sink) -> None:
        self.queue = queue
        self.sink = sink

    def run(self):
        while True:
            obj = self.queue.get()
            if obj is None:
                self.sink.close()
                return
            
            self.sink.put(obj)


def decode_line(line):
    orig = json.loads(line)
    out = {
        'id': orig['identifier'],
        'title': orig['name'],
        'url': orig['url'],
        'text': orig['article_body']['wikitext'],
        'html': orig['article_body']['html'],
    }
    return out


def extract_plain(obj):
    text = obj['text']
    plain = parse_and_clean_wikitext(text)
    obj['plain'] = plain
    return obj


def extract_worker(inq, outq):
    while True:
        obj = inq.get()
        if obj is False:
            outq.put(None)
            outq.close()
            outq.join_thread()
            return
        
        obj = extract_plain(obj)
        outq.put(obj)


def scan_dump_lines(dump_file: Path):
    with tarfile.open(dump_file, 'r:gz') as tf:
        ti = tf.next()
        while ti is not None:
            _log.info('reading tar member %s', ti.name)
            with tqdm(total=ti.size, desc='bytes', unit='B', unit_scale=True, leave=False) as pbar:
                with tf.extractfile(ti) as jsf:
                    for line in jsf:
                        pbar.update(len(line))
                        yield line
                        
            ti = tf.next()


def main(opts):
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

    file = Path(opts['DUMP'])
    nthreads = opts.get('-j', None)
    if nthreads is None:
        nthreads = cpu_count() - 4
    else:
        nthreads = int(nthreads)
    
    _log.info('opening output files')
    sinks = [
        Sink('data/trec_corpus_html.json.gz', 'html'),
        Sink('data/trec_corpus_text.json.gz', 'text'),
        Sink('data/trec_corpus_plain.json.gz', 'plain'),
    ]

    for line in scan_dump_lines(file):
        obj = decode_line(line)
        obj = extract_plain(obj)
        for sink in sinks:
            sink.put(obj)
    
    _log.info('finished, closing sinks')
    for sink in sinks:
        sink.close()


if __name__ == '__main__':
    opts = docopt(__doc__)
    main(opts)
