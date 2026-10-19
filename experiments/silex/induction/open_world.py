"""Exploratory falsification: consensus under an inadequate program language.
Mixed policy spends at most three labels: one disagreement + two coverage audits.
The learner never sees the teacher implementation. No target-specific DSL patch.
"""
import json
from pathlib import Path
import random
import resource
import tempfile
from core import Space,key
from probes import generate
from bench import evaluate

TEACHERS={
 'upper_clamp_10':lambda x:min(x,10),
 'lower_clamp_minus10':lambda x:max(x,-10),
 'shift_outside_20':lambda x:x if abs(x)<=20 else x+1,
 'exception_at_137':lambda x:x+1 if x==137 else x,
 'sign_flip_above10':lambda x:x if x<=10 else -x,
}
HELDOUT=[-250,-137,-31,-17,-7,0,6,9,13,22,31,137,250]


def run():
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        for name,teacher in TEACHERS.items():
            for seed in (7,17,27):
                pool=[x for x in generate('int',seed) if x not in HELDOUT+[-1,1,2]]
                for policy in ('active','active_plus_audit','random'):
                    s=Space(Path(tmp)/f'{name}-{seed}-{policy}.sqlite')
                    s.fit([dict(input=x,output=teacher(x)) for x in (-1,1,2)])
                    rng=random.Random(seed);history=[]
                    for turn in range(3):
                        if not s.count():break
                        seen=[e['input'] for e in s.get('examples')]
                        available=[x for x in pool if x not in seen]
                        mode='active' if policy=='active' or (policy=='active_plus_audit' and turn==0) else 'audit' if policy=='active_plus_audit' else 'random'
                        if mode=='active':
                            q=s.clarify(available)
                            if q['status']!='question':
                                if policy=='active':break
                                continue
                            x=q['input']
                        elif mode=='audit':
                            x=max(available,key=lambda v:min(abs(v-o) for o in seen))
                        else:x=rng.choice(available)
                        y=teacher(x);s.observe(x,y)
                        history.append(dict(mode=mode,input=x,output=y,hypotheses=s.count()))
                    rows.append(dict(teacher=name,seed=seed,policy=policy,queries=history,
                                     evaluation=evaluate(s,HELDOUT,[teacher(x) for x in HELDOUT])))
                    s.close()
    return dict(exploratory=True,heldout=HELDOUT,teacher_in_DSL=False,max_additional_labels=3,
                results=rows,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                address_space_limit_bytes=100000000)

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(100000000,100000000))
    print(json.dumps(run(),ensure_ascii=False,indent=2))
