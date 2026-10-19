"""Broader synthetic stress suite; NOT independent human language evaluation.
Protocol: 256 distinct (domain, target syntax) cases, seed 20261019;
lengths 1..4; one initial pair, at most three further labels, 8 held-out inputs.
Depth-3 hypothesis space. Two policies share the same inputs and observations.
Target validity filtering is performed by the evaluator, never by the learner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import tempfile
import time
from core import Space,OPS,INVALID,execute,key
from probes import generate
from bench import evaluate


def cases(count=256,seed=20261019):
    rng=random.Random(seed);seen=set();out=[]
    for index in range(count):
        kind=('int','text','list')[index%3]
        pool=generate(kind,seed+index,48)
        forbidden={key(x) for x in pool}
        heldout=[x for x in generate(kind,seed+10000+index) if key(x) not in forbidden][:8]
        for _ in range(100000):
            target=[rng.choice(list(OPS)) for _ in range(rng.randint(1,4))]
            signature=key([kind,target])
            if signature in seen:continue
            outputs=[execute(target,x) for x in pool+heldout]
            if any(y is INVALID for y in outputs):continue
            seen.add(signature);out.append(dict(index=index,kind=kind,target=target,pool=pool,heldout=heldout));break
        else:raise RuntimeError('case generation budget exhausted')
    return out


def run(count,seed):
    tasks=cases(count,seed);results=[];start=time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for task in tasks:
            i=task['index'];target=task['target'];pool=task['pool'];tests=task['heldout']
            truth=[execute(target,x) for x in tests]
            base=Space(Path(tmp)/'base.sqlite')
            search=base.fit([dict(input=pool[0],output=execute(target,pool[0]))])
            row=dict(index=i,kind=task['kind'],target=target,search=search,
                     initial=evaluate(base,tests,truth),policies=[])
            for policy in ('active','random'):
                path=Path(tmp)/f'{policy}.sqlite';base.copy_to(path);s=Space(path)
                available=pool[1:].copy();rng=random.Random(seed+i);history=[]
                for _ in range(3):
                    if not s.count():break
                    if policy=='active':
                        question=s.clarify(available)
                        if question['status']!='question':break
                        x=question['input']
                    else:x=rng.choice(available)
                    available=[v for v in available if key(v)!=key(x)]
                    answer=execute(target,x);s.observe(x,answer)
                    history.append(dict(input=x,output=answer,remaining=s.count()))
                row['policies'].append(dict(policy=policy,queries=history,final=evaluate(s,tests,truth)))
                s.close()
            base.close();results.append(row)
            if (i+1)%32==0:print(f'{i+1}/{count} cases complete',flush=True)
    return dict(seed=seed,count=count,generated_suite_sha256=hashlib.sha256(key(tasks).encode()).hexdigest(),
                elapsed_seconds=time.perf_counter()-start,results=results,
                rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                address_space_limit_bytes=100000000,caveat='Synthetic tasks in/near our DSL, not blind or general language.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261019);p.add_argument('--output',required=True)
    a=p.parse_args();resource.setrlimit(resource.RLIMIT_AS,(100000000,100000000))
    result=run(a.count,a.seed);Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Completed:',a.output,flush=True)
