"""Fresh-process depth scaling and honest refusal at the search budget."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from core import Space
from probes import generate


def worker(depth):
    resource.setrlimit(resource.RLIMIT_AS,(100000000,100000000))
    with tempfile.TemporaryDirectory() as tmp:
        s=Space(Path(tmp)/'space.sqlite');start=time.perf_counter()
        fit=s.fit([dict(input=1,output=1)],depth=depth,max_nodes=100000,max_matches=10000)
        fit_seconds=time.perf_counter()-start
        start=time.perf_counter();q=s.clarify(generate('int',7));query_seconds=time.perf_counter()-start
        out=dict(depth=depth,search=fit,fit_seconds=fit_seconds,query_seconds=query_seconds,
                 question=q,prediction=s.predict(-2),sqlite_bytes=s.path.stat().st_size,
                 rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 address_space_limit_bytes=100000000)
        s.close();return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--depth',type=int);a=p.parse_args()
    if a.depth is not None:result=worker(a.depth)
    else:result=dict(results=[json.loads(subprocess.check_output([sys.executable,__file__,'--depth',str(d)]))
                             for d in range(1,7)],fresh_process_per_depth=True)
    print(json.dumps(result,ensure_ascii=False,indent=2))
