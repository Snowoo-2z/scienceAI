"""Exploratory follow-up after fixed-pool results, not a new blind benchmark.
Tests generated queries and deliberately misspecified hypothesis classes.
"""
import json
from pathlib import Path
import random
import resource
import tempfile
from core import Space,execute,key
from tasks import TASKS,TEST
from bench import evaluate
from probes import generate


def run():
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        for name,kind,target,initial in TASKS:
            examples=[dict(input=x,output=execute(target,x)) for x in initial]
            expected=[execute(target,x) for x in TEST[kind]]
            base=Space(Path(tmp)/f'{name}.sqlite');base.fit(examples)
            for seed in (7,17,27):
                # Remove exact collisions with held-out inputs, without their labels.
                excluded={key(x) for x in TEST[kind]+initial}
                pool=[x for x in generate(kind,seed) if key(x) not in excluded]
                for policy in ('active_generated','random_generated'):
                    path=Path(tmp)/f'{name}-{seed}-{policy}.sqlite';base.copy_to(path);s=Space(path)
                    rng=random.Random(seed);available=pool.copy();history=[]
                    for _ in range(3):
                        if policy=='active_generated':
                            q=s.clarify(available)
                            if q['status']!='question':break
                            x=q['input']
                        else:
                            if not available:break
                            x=rng.choice(available);q=dict(input=x)
                        available=[v for v in available if key(v)!=key(x)]
                        answer=execute(target,x);s.observe(x,answer)
                        history.append(dict(input=x,output=answer,hypotheses=s.count()))
                    results.append(dict(task=name,seed=seed,policy=policy,queries=history,
                                        final=evaluate(s,TEST[kind],expected)))
                    s.close()
            base.close()
        # A hidden upper clamp is NOT representable in the depth-3 DSL.
        # Identity-like hypotheses fit observations and can agree on wrong outputs.
        s=Space(Path(tmp)/'misspecified.sqlite')
        teacher=lambda x:min(x,10)
        s.fit([dict(input=x,output=teacher(x)) for x in (-1,1,2)])
        query_history=[]
        for _ in range(3):
            q=s.clarify(list(range(-9,12)))
            if q['status']!='question':break
            x=q['input'];s.observe(x,teacher(x));query_history.append(dict(input=x,output=teacher(x)))
        tests=[-17,-7,6,9,13,22,31]
        before=evaluate(s,tests,[teacher(x) for x in tests])
        details=[dict(input=x,expected=teacher(x),prediction=s.predict(x)) for x in tests]
        # An audit query is not selected from disagreement. It exposes model error.
        s.observe(20,teacher(20));after=evaluate(s,tests,[teacher(x) for x in tests])
        stress=dict(teacher='min(x,10), evaluator only, not in DSL',queries=query_history,
                    before_audit=before,details=details,audit=dict(input=20,output=10),after_audit=after)
        s.close()
    return dict(exploratory=True,generated_queries=results,misspecification=stress,
                rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(100000000,100000000))
    print(json.dumps(run(),ensure_ascii=False,indent=2))
