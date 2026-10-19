"""S6: learned instruction routing -> learned disk-backed operators.
Restricted controlled-French task. No conversational model or novelty claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
import unicodedata
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paged'))
from library import Paged,connect,independent
from data import dataset

FEATURES=512
REG=1e-4


def features(texts):
    x=np.zeros((len(texts),FEATURES),np.float32)
    for i,text in enumerate(texts):
        s=' '+unicodedata.normalize('NFKC',text).casefold()+' '
        for n in (2,3,4):
            for j in range(len(s)-n+1):
                code=int.from_bytes(hashlib.blake2b(s[j:j+n].encode(),digest_size=8).digest(),'little')
                x[i,code%FEATURES]+=1 if code>>63 else -1
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    return x


def softmax(z):
    z=z-z.max(axis=1,keepdims=True)
    p=np.exp(z);return p/p.sum(axis=1,keepdims=True)


def loss_grad(x,y,w,b):
    p=softmax(x@w+b)
    loss=float(-np.log(p[np.arange(len(y)),y]+1e-30).mean()+REG/2*np.sum(w*w))
    g=p.copy();g[np.arange(len(y)),y]-=1;g/=len(y)
    return loss,x.T@g+REG*w,g.sum(axis=0)


def train(rows,seed,reject_class,steps=500):
    rows=[r for r in rows if reject_class or r['label']>=0]
    x=features([r['text'] for r in rows]);y=np.array([r['label'] if r['label']>=0 else 8 for r in rows])
    rng=np.random.default_rng(seed)
    w=(rng.normal(size=(FEATURES,9 if reject_class else 8))*.01).astype(np.float32)
    b=np.zeros(w.shape[1],np.float32)
    m=[np.zeros_like(w),np.zeros_like(b)];v=[a.copy() for a in m]
    first=None
    for step in range(1,steps+1):
        loss,dw,db=loss_grad(x,y,w,b)
        if first is None:first=loss
        for i,(a,g) in enumerate(((w,dw),(b,db))):
            m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
            a-=.03*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
    return (w,b),dict(training_rows=len(rows),initial_loss=first,final_loss=loss_grad(x,y,w,b)[0],
                      parameter_bytes=w.nbytes+b.nbytes)


def predict(model,x):
    p=softmax(x@model[0]+model[1]);labels=p.argmax(axis=1)
    scores=p[np.arange(len(x)),labels]
    return np.where(labels==8,-1,labels),scores


def calibrate(labels,scores,truth,max_risk=.05,min_accepted=16):
    # Maximize empirical coverage under an error limit on VALIDATION only.
    # This is not a statistical guarantee of risk under distribution shift.
    best=dict(threshold=1.000001,accepted=0,wrong=0,empirical_risk=None)
    for threshold in sorted(set(map(float,scores))):
        accept=(labels>=0)&(scores>=threshold)
        n=int(accept.sum());wrong=int(((labels!=truth)&accept).sum())
        if n>=min_accepted and wrong/n<=max_risk and n>best['accepted']:
            best=dict(threshold=threshold,accepted=n,wrong=wrong,empirical_risk=wrong/n)
    return best


def metrics(rows,pred,scores,threshold):
    y=np.array([r['label'] for r in rows]);known=y>=0
    accept=(pred>=0)&(scores>=threshold)
    correct=accept&(pred==y);wrong=accept&(pred!=y)
    return dict(count=len(rows),coverage=float(accept.mean()),
                supported_route_accuracy_before_threshold=float((pred[known]==y[known]).mean()),
                supported_correct_fraction=float(correct[known].mean()),
                unsupported_false_accept_fraction=float(accept[~known].mean()),
                accepted_accuracy=float(correct.sum()/accept.sum()) if accept.any() else None,
                wrong_accepted=int(wrong.sum()),accepted=int(accept.sum()),
                details=[dict(**r,predicted_label=int(a),score=float(s),accepted=bool(ok))
                         for r,a,s,ok in zip(rows,pred,scores,accept)])


def target(op,d=16):
    t=np.eye(d);t[[op,op+1]]=t[[op+1,op]];return t


def build_library(root,seed):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    c=connect(root/'adaptive.sqlite',True)
    for op in range(8):
        rng=np.random.default_rng(seed+100+op)
        x=rng.normal(size=(32,16));y=x@target(op).T
        learned=np.linalg.lstsq(x,y,rcond=None)[0].T
        xv=rng.normal(size=(48,16));yv=xv@target(op).T
        blob=independent(learned,xv,yv,1e-5)
        c.execute('INSERT INTO operators VALUES (?,?)',(op,blob))
    c.commit();c.close()
    (root/'metadata.json').write_text(json.dumps(dict(d=16,count=8)))


def programs(model,rows,threshold,root,budget,seed,policy,program_count=128,route=None):
    supported=[r for r in rows if r['label']>=0]
    rng=np.random.default_rng(seed+5000)
    resident=0 if policy=='oracle' else sum(a.nbytes for a in model)
    if resident>budget:raise ValueError('router exceeds total payload budget')
    paged=Paged(root,'adaptive',budget-resident)
    result=[]
    for length in (1,4,16,64):
        completed=correct=bad_routes=used=0;latencies=[]
        before=paged.logical_bytes
        for _ in range(program_count):
            instructions=[supported[int(i)] for i in rng.integers(len(supported),size=length)]
            expected=np.eye(16)
            for row in instructions:expected=expected@target(row['label']).T
            actual=np.eye(16);abstained=False
            start=time.perf_counter()
            for row in instructions:
                if policy=='oracle':selected=row['label']
                else:
                    pred,conf=route([row['text']]) if route is not None else predict(model,features([row['text']]))
                    selected=int(pred[0])
                    if selected<0 or (policy=='selective' and conf[0]<threshold):
                        abstained=True;break
                used+=1;bad_routes+=int(selected!=row['label'])
                actual=paged.apply(actual,selected)
            latencies.append(time.perf_counter()-start)
            if not abstained:
                completed+=1
                correct+=int(np.linalg.norm(actual-expected)/np.linalg.norm(expected)<1e-4)
        result.append(dict(length=length,programs=program_count,completed=completed,correct=correct,
                           completion_fraction=completed/program_count,correct_fraction=correct/program_count,
                           correct_given_completion=correct/completed if completed else None,
                           applied_steps=used,incorrect_step_routes=bad_routes,
                           logical_read_bytes=paged.logical_bytes-before,p50_ms=float(np.percentile(latencies,50)*1000),
                           p95_ms=float(np.percentile(latencies,95)*1000),
                           router_resident_bytes=resident,operator_cache_capacity=paged.capacity,
                           peak_router_plus_operator_payload=resident+paged.peak))
    paged.close()
    return result


def run(seed,steps=500):
    data=dataset();rows=[]
    for reject in (False,True):
        start=time.perf_counter();model,training=train(data['train'],seed,reject,steps)
        train_seconds=time.perf_counter()-start
        vy=np.array([r['label'] for r in data['validation']])
        vp,vs=predict(model,features([r['text'] for r in data['validation']]))
        calibration=calibrate(vp,vs,vy)
        tp,ts=predict(model,features([r['text'] for r in data['test']]))
        with tempfile.TemporaryDirectory() as root:
            build_library(root,seed)
            compositions={policy:programs(model,data['test'],calibration['threshold'],root,65536,seed,policy)
                          for policy in ('oracle','top1','selective')}
        rows.append(dict(seed=seed,reject_class=reject,training=training,training_seconds=train_seconds,
                         calibration=calibration,
                         top1_test=metrics(data['test'],tp,ts,0),
                         selective_test=metrics(data['test'],tp,ts,calibration['threshold']),
                         compositions=compositions))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--steps',type=int,default=500)
    a=p.parse_args()
    if a.steps<1:p.error('steps must be positive')
    rows=[]
    for seed in a.seeds:rows.extend(run(seed,a.steps))
    print(json.dumps(dict(date='2026-10-19',model='hashed_ngram_linear_router_to_learned_operators',
                         numpy=np.__version__,python=platform.python_version(),hardware=platform.machine(),
                         quantization='none_float32_router_float64_operators',
                         context='controlled_French_instruction_list_with_given_step_boundaries',
                         steps=a.steps,seeds=a.seeds,
                         data_sha256=hashlib.sha256(json.dumps(dataset(),sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                         rss_scope='training_and_evaluation_combined_not_deployed_service',
                         rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                         energy=None,language_benchmark=None,results=rows),ensure_ascii=False,indent=2))
