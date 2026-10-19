"""Diagnostic ablation: learned binary intent + SUPPLIED numeric binding.
Numeric recognition/domain checks are handwritten, not learned reasoning.
Fresh challenge templates are separate from S6 first-pass test templates.
"""
import argparse
import hashlib
import json
import re
import resource
import tempfile
import time
import numpy as np
from experiment import features,loss_grad,softmax,calibrate,metrics,programs,build_library
from data import dataset,expand

NUMBERS={w:i+1 for i,w in enumerate(['un','deux','trois','quatre','cinq','six','sept','huit','neuf',
                                   'dix','onze','douze','treize','quatorze','quinze','seize'])}
PATTERN=re.compile(r'\b(?:\d+|'+'|'.join(NUMBERS)+r')\b',re.IGNORECASE)


def arguments(text):
    values=[int(m) if m.isdecimal() else NUMBERS[m.casefold()] for m in PATTERN.findall(text)]
    if len(values)<2:return None
    a,b=sorted(values[-2:])
    return a-1 if b==a+1 and 1<=a<b<=9 else None


def masked(text):return PATTERN.sub(' # ',text)


def fit(rows,seed,steps=500):
    x=features([masked(r['text']) for r in rows])
    y=np.array([int(r['label']>=0) for r in rows])
    rng=np.random.default_rng(seed)
    w=(rng.normal(size=(x.shape[1],2))*.01).astype(np.float32);b=np.zeros(2,np.float32)
    m=[np.zeros_like(w),np.zeros_like(b)];v=[a.copy() for a in m]
    for step in range(1,steps+1):
        loss,dw,db=loss_grad(x,y,w,b)
        for i,(a,g) in enumerate(((w,dw),(b,db))):
            m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
            a-=.03*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
    return (w,b)


def route(model,texts):
    p=softmax(features([masked(t) for t in texts])@model[0]+model[1])
    arg=[arguments(t) for t in texts]
    labels=np.array([a if a is not None and probability[1]>=probability[0] else -1
                     for a,probability in zip(arg,p)])
    return labels,p[:,1]


def challenge():
    positive=expand(['échange {a} avec {b} dans cette suite',
                     "mets l'élément {a} à la place de {b} et celui de {b} à la place de {a}"])
    negative=expand(['tu ne dois pas intervertir les positions {a} et {b}',
                     'explique comment échanger {a} et {b}, mais sans rien modifier',
                     'ne touche à rien, même si je mentionne les positions {a} et {b}'],True)
    extra=[dict(text=t,label=-1,kind='unsupported',spelling=None) for t in [
        'échange 1 et 2 puis échange 3 et 4',
        'intervertis les trois positions 1, 2 et 3',
        'échange la première et la dernière position',
        'fais trois échanges aléatoires',
        'pourquoi faudrait-il permuter 4 et 5 ?',
        'si la liste est vide, échange 2 et 3']]
    return positive+negative+extra


def run(seed):
    data=dataset();start=time.perf_counter();model=fit(data['train'],seed)
    elapsed=time.perf_counter()-start
    vp,vs=route(model,[r['text'] for r in data['validation']])
    calibration=calibrate(vp,vs,np.array([r['label'] for r in data['validation']]))
    result=dict(seed=seed,parameter_bytes=sum(a.nbytes for a in model),training_seconds=elapsed,
                calibration=calibration)
    for name,rows in [('diagnostic_test',data['test']),('new_challenge',challenge())]:
        pred,conf=route(model,[r['text'] for r in rows])
        record={policy:metrics(rows,pred,conf,0 if policy=='top1' else calibration['threshold'])
                for policy in ('top1','selective')}
        with tempfile.TemporaryDirectory() as root:
            build_library(root,seed)
            record['programs']={policy:programs(model,rows,calibration['threshold'],root,65536,seed,policy,
                                                 route=lambda texts:route(model,texts))
                                for policy in ('oracle','top1','selective')}
        result[name]=record
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    a=p.parse_args()
    rows=[run(s) for s in a.seeds]
    print(json.dumps(dict(date='2026-10-19',model='binary_intent_plus_supplied_numeric_binding',
                         caveat='Handwritten arguments and supplied sequence boundaries; NOT an end-to-end learned language parser.',
                         challenge_sha256=hashlib.sha256(json.dumps(challenge(),sort_keys=True).encode()).hexdigest(),
                         rss_scope='training_and_evaluation_combined_not_deployed_service',
                         rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,results=rows),
                     ensure_ascii=False,indent=2))
