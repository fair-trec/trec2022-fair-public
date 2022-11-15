import logging
from pathlib import Path
import gzip
import re
import pandas as pd

_log = logging.getLogger(__name__)


def scan_teams(dir = 'runs'):
    path = Path(dir) / 'metadata'
    run_re = re.compile(r'^Run (.*):')
    records = []
    for file in path.glob('confirm.fair.*'):
        team = file.suffix[1:]
        with file.open('r', encoding='utf8') as mf:
            for line in mf:
                m = run_re.match(line)
                if m:
                    records.append((team, m.group(1)))
    
    return pd.DataFrame.from_records(records, columns=['team', 'run'])


def scan_runs(task, dir='runs'):
    dn = 'coordinators' if task == 1 else 'editors'
    path = Path(dir) / dn
    for file in path.glob('*.gz'):
        _log.info('scanning %s', file)
        run_name = file.stem
        rank = 1
        prev_topic_id = None
        prev_seq = None
        with gzip.open(file, 'rt') as gzf:
            rows=[]
            for lno, line in enumerate(gzf):
                line = line.strip().split('\t')
                if line[0] == 'id':
                    continue  # header row
                
                if task == 1:
                    topic_id = int(line[0])
                    page_id = int(line[1])
                    if topic_id == prev_topic_id:
                        rank += 1
                    else:
                        rank = 1
                        prev_topic_id = topic_id
                    
                    rows.append({
                        'run_name': run_name,
                        'topic_id': topic_id,
                        'rank': rank,
                        'page_id': page_id
                    })
                
                else:
                    topic_id = int(line[0])
                    seq_no = int(line[1])
                    page_id = int(line[2])
                    if prev_topic_id == topic_id and prev_seq == seq_no:
                        rank+=1
                    else:
                        rank=0
                        prev_topic_id = topic_id
                        prev_seq = seq_no
                    rows.append({
                        'run_name': run_name,
                        'topic_id': topic_id,
                        'seq_no': seq_no,
                        'rank': rank,
                        'page_id': page_id
                    })

            yield rows