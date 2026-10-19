"""Build and run each runtime configuration in a separate process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--cold-file',action='store_true')
p.add_argument('--root',default='.cache/silex-s5')
p.add_argument('--output',default='experiments/silex/paged/runtime-results.json')
p.add_argument('--requests',type=int,default=3000)
a=p.parse_args()
script=Path(__file__).with_name('bench.py')
env={**os.environ,'OPENBLAS_NUM_THREADS':'1'}
rows=[]
for method in ('dense','adaptive','shared'):
    for budget in (65536,262144):
        for workload in ('uniform','zipf'):
            command=[sys.executable,str(script),'bench','--root',a.root,'--method',method,
                     '--budget',str(budget),'--workload',workload,'--requests',str(a.requests)]
            if a.cold_file:command.append('--cold-file')
            row=json.loads(subprocess.check_output(command,env=env,text=True))
            row['command']=command
            rows.append(row)
Path(a.output).write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
    print(r['method'],r['budget_payload_bytes'],r['workload'],
          'hit',round(r['hits']/r['requests'],3),'logical',r['logical_payload_read_bytes'],
          'p95',round(r['p95_ms'],3),'RSS',r['rss_peak_bytes'],'pass',r['test_pass_fraction'])
