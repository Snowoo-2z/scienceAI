import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import tempfile
import time
from core import Space,execute,INVALID,key
from tasks import TASKS,POOLS,TEST


def evaluate(space,values,truth):
    answers=correct=wrong=short_correct=0;statuses={}
    for x,y in zip(values,truth):
        p=space.predict(x);statuses[p['status']]=statuses.get(p['status'],0)+1
        if p['status']=='consensus':
            answers+=1;correct+=int(key(p['output'])==key(y));wrong+=int(key(p['output'])!=key(y))
        s=space.shortest(x)
        short_correct+=int(s['status']=='answer' and key(s['output'])==key(y))
    return dict(tests=len(values),answers=answers,correct=correct,wrong=wrong,
                shortest_correct=short_correct,statuses=statuses,hypotheses=space.count())


def run(rounds=3,seeds=(7,17,27)):
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        for name,kind,target,initial in TASKS:
            expected=[execute(target,x) for x in TEST[kind]]
            assert all(v is not INVALID for v in expected)
            examples=[dict(input=x,output=execute(target,x)) for x in initial]
            base=Space(Path(tmp)/f'{name}.sqlite');start=time.perf_counter()
            search=base.fit(examples,depth=3)
            record=dict(task=name,kind=kind,evaluator_only_target=target,examples=examples,
                        search=search,search_seconds=time.perf_counter()-start,
                        initial=evaluate(base,TEST[kind],expected),policies=[])
            for policy in ('active','random'):
                for seed in (seeds if policy=='random' else [seeds[0]]):
                    path=Path(tmp)/f'{name}-{policy}-{seed}.sqlite';base.copy_to(path);space=Space(path)
                    rng=random.Random(seed);available=POOLS[kind].copy();history=[]
                    start=time.perf_counter()
                    for turn in range(rounds):
                        if policy=='active':
                            question=space.clarify(available)
                            if question['status']!='question':break
                            x=question['input']
                        else:
                            seen={key(e['input']) for e in space.get('examples')}
                            options=[x for x in available if key(x) not in seen]
                            if not options:break
                            x=rng.choice(options);question=dict(status='question',input=x)
                        answer=execute(target,x)
                        assert answer is not INVALID
                        space.observe(x,answer)
                        history.append(dict(question=question,teacher_output=answer,
                                            evaluation=evaluate(space,TEST[kind],expected)))
                    record['policies'].append(dict(policy=policy,seed=seed,history=history,
                         seconds=time.perf_counter()-start,final=evaluate(space,TEST[kind],expected),
                         sqlite_bytes=path.stat().st_size))
                    space.close()
            record['sqlite_bytes']=base.path.stat().st_size;base.close();results.append(record)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--address-space-mb',type=int,default=100)
    a=p.parse_args()
    if a.address_space_mb:
        resource.setrlimit(resource.RLIMIT_AS,(a.address_space_mb*1000000,)*2)
    begin=time.perf_counter();rows=run(a.rounds)
    payload=dict(date='2026-10-19',python=platform.python_version(),machine=platform.machine(),
        language_model=False,primitives='handwritten_allowlist',search_depth=3,
        dataset_sha256=hashlib.sha256(json.dumps([TASKS,POOLS,TEST],sort_keys=True).encode()).hexdigest(),
        address_space_limit_bytes=a.address_space_mb*1000000,
        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        rss_scope='search_and_evaluation_combined',elapsed_seconds=time.perf_counter()-begin,
        energy=None,results=rows)
    print(json.dumps(payload,ensure_ascii=False,indent=2))
