"""S4: learned transformations, not stored question/answer pairs.
Synthetic mechanism benchmark. Instruction parser/composition loop are supplied.
No language understanding or novelty claim. NumPy only.
"""
import argparse
import json
import platform
from pathlib import Path
import resource
import time
import numpy as np

ROOT=Path(__file__).resolve().parent


def targets(family, d, k, seed):
    rng=np.random.default_rng(seed)
    t=np.repeat(np.eye(d)[None,:,:],k,axis=0)
    if family=='swaps':
        for i in range(k):
            a=i % (d-1)
            t[i,[a,a+1]]=t[i,[a+1,a]]
    elif family=='random_permutations':
        for i in range(k):t[i]=np.eye(d)[rng.permutation(d)]
    elif family=='planted_shared':
        # Positive control deliberately constructed from a low-rank dictionary.
        r=4
        u=rng.normal(size=(d,r))/np.sqrt(d)
        v=rng.normal(size=(r,d))/np.sqrt(d)
        c=rng.normal(size=(k,r))*.15
        t+=np.einsum('dr,kr,rs->kds',u,c,v)
    else:raise ValueError(family)
    return t


def init(d,k,r,seed):
    rng=np.random.default_rng(seed)
    return dict(U=rng.normal(size=(d,r))*.15,
                V=rng.normal(size=(r,d))*.15,
                C=rng.normal(size=(k,r))*.15)


def loss_grad(p,x,y):
    z=x@p['V'].T
    a=z*p['C'][:,None,:]
    pred=x+a@p['U'].T
    e=pred-y
    loss=float(np.mean(e**2))
    e*=2/e.size
    ga=e@p['U']
    return loss,dict(U=np.einsum('knd,knr->dr',e,a),
                     V=np.einsum('knr,knd->rd',ga*p['C'][:,None,:],x),
                     C=np.sum(ga*z,axis=1))


def fit_shared(x,y,rank,seed,steps,anneal=False):
    k,_,d=x.shape
    p=init(d,k,rank,seed)
    m={a:np.zeros_like(b) for a,b in p.items()}
    v={a:np.zeros_like(b) for a,b in p.items()}
    initial=None
    for step in range(1,steps+1):
        loss,g=loss_grad(p,x,y)
        if initial is None:initial=loss
        for a in p:
            m[a]=.9*m[a]+.1*g[a]
            v[a]=.999*v[a]+.001*g[a]*g[a]
            lr=.025 * (max(.001, (steps-step)/max(steps/2,1)) if anneal and step>steps/2 else 1.)
            p[a]-=lr*(m[a]/(1-.9**step))/(np.sqrt(v[a]/(1-.999**step))+1e-8)
    return p,dict(initial_mse=initial,final_train_mse=loss_grad(p,x,y)[0])


def apply(model,x,op):
    method,p=model
    if method=='shared':
        return x+((x@p['V'].T)*p['C'][op])@p['U'].T
    if method=='independent_rank1':
        return x+(x@p['V'][op][:,None])*p['U'][op][None,:]
    if method=='dense':return x@p['T'][op].T
    if method=='identity':return x.copy()
    raise ValueError(method)


def fit_controls(x,y):
    k,_,d=x.shape
    # Recover from paired examples, not from ground-truth target matrices.
    matrices=np.array([np.linalg.lstsq(x[i],y[i],rcond=None)[0].T for i in range(k)])
    us=[];vs=[]
    for t in matrices:
        u,s,v=np.linalg.svd(t-np.eye(d),full_matrices=False)
        us.append(u[:,0]*s[0]);vs.append(v[0])
    return [('dense',dict(T=matrices)),('independent_rank1',dict(U=np.array(us),V=np.array(vs))),('identity',{})]


def evaluate(model,truth,length,seed,trials=128):
    rng=np.random.default_rng(seed)
    k,d,_=truth.shape
    errors=[]
    complete=[]
    examples=[]
    for trial in range(trials):
        ops=rng.integers(k,size=length).tolist()
        # All basis probes: evaluates the complete composed linear operator.
        expected=np.eye(d)
        actual=np.eye(d)
        for op in ops:
            expected=expected@truth[op].T
            actual=apply(model,actual,op)
        rel=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12))
        errors.append(rel)
        complete.append(rel<1e-3)
        if trial==0:
            examples.append(dict(operators=ops,relative_error=rel,
                                 expected_first_row=expected[0].tolist(),actual_first_row=actual[0].tolist()))
    return dict(length=length,mean_relative_error=float(np.mean(errors)),
                complete_operator_fraction_under_1e_3=float(np.mean(complete)),example=examples[0])


def run(seed,steps,anneal=False):
    result=[]
    d,k,rank,n=16,8,8,32
    for family in ('swaps','random_permutations','planted_shared'):
        truth=targets(family,d,k,seed+200)
        rng=np.random.default_rng(seed+100)
        x=rng.normal(size=(k,n,d))
        y=np.einsum('knd,ked->kne',x,truth)
        start=time.perf_counter()
        shared,losses=fit_shared(x,y,rank,seed,steps,anneal)
        training_seconds=time.perf_counter()-start
        models=[('shared',shared)]+fit_controls(x,y)
        for method,p in models:
            begin=time.perf_counter()
            scores=[evaluate((method,p),truth,length,seed+1000+length) for length in (1,4,16,64)]
            result.append(dict(family=family,method=method,seed=seed,
                               parameters=sum(a.size for a in p.values()),
                               payload_bytes=sum(a.nbytes for a in p.values()),
                               training_seconds_shared_only=training_seconds if method=='shared' else None,
                               evaluation_seconds=time.perf_counter()-begin,
                               losses=losses if method=='shared' else None,scores=scores))
    return result


def tests():
    rng=np.random.default_rng(42)
    p=init(4,3,2,42)
    x=rng.normal(size=(3,5,4));y=rng.normal(size=x.shape)
    _,g=loss_grad(p,x,y)
    checks=0
    for key in p:
        for index in np.ndindex(p[key].shape):
            old=p[key][index]
            p[key][index]=old+1e-5;a,_=loss_grad(p,x,y)
            p[key][index]=old-1e-5;b,_=loss_grad(p,x,y)
            p[key][index]=old
            assert abs((a-b)/2e-5-g[key][index])<1e-7
            checks+=1
    dense=np.eye(4)+p['U']@np.diag(p['C'][1])@p['V']
    np.testing.assert_allclose(apply(('shared',p),x[0],1),x[0]@dense.T,atol=1e-12)
    truth=targets('swaps',4,3,1)
    assert not np.array_equal(truth[0]@truth[1],truth[1]@truth[0])
    for length in (1,4,16):
        assert evaluate(('dense',{'T':truth}),truth,length,42,8)['complete_operator_fraction_under_1e_3']==1
    print(checks,'gradient checks, factorized equivalence, noncommuting order and oracle composition PASS')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anneal',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=parser.parse_args()
    if args.test:tests();return
    if args.steps<1:parser.error('steps must be >= 1')
    rows=[]
    for seed in args.seeds:
        rows.extend(run(seed,args.steps,args.anneal))
    print(json.dumps(dict(model='residual_shared_operator_dictionary',date='2026-10-19',
                          quantization='none_float64',backend='numpy_'+np.__version__,
                          python=platform.python_version(),hardware=platform.machine(),
                          context='symbolic operator sequence; no natural-language parser',
                          d=16,k=8,rank=8,training_examples_per_operator=32,steps=args.steps,anneal=args.anneal,
                          rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          energy=None,language_quality=None,rows=rows),indent=2))


if __name__=='__main__':main()
