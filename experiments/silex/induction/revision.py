"""Exploratory depth widening after failed fit, with unchanged primitive set."""
import json
from pathlib import Path
import resource
import tempfile
import time
from core import Space,execute,key
from probes import generate
from tasks import TEST,TASKS
from bench import evaluate


def run():
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        name,kind,target,initial=TASKS[-1]
        for seed in (7,17,27):
            s=Space(Path(tmp)/f'{seed}.sqlite');s.fit([dict(input=x,output=execute(target,x)) for x in initial])
            start=time.perf_counter();revised=s.revise_if_empty(max_depth=4)
            excluded={key(x) for x in initial+TEST[kind]}
            pool=[x for x in generate(kind,seed) if key(x) not in excluded];history=[]
            for _ in range(3):
                q=s.clarify(pool)
                if q['status']!='question':break
                x=q['input'];y=execute(target,x);s.observe(x,y)
                history.append(dict(input=x,output=y,hypotheses=s.count()))
            rows.append(dict(seed=seed,task=name,revision=revised,queries=history,
                             evaluation=evaluate(s,TEST[kind],[execute(target,x) for x in TEST[kind]]),
                             elapsed_seconds=time.perf_counter()-start,sqlite_bytes=s.path.stat().st_size))
            s.close()
    return dict(results=rows,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                address_space_limit_bytes=100000000,exploratory=True)

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(100000000,100000000))
    print(json.dumps(run(),ensure_ascii=False,indent=2))
