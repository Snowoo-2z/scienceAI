"""S5: validated mixed operator representations, bounded payload LRU, SQLite backing.
Synthetic learned operators; addresses supplied by benchmark, NOT a language model.
"""
from collections import OrderedDict
import json
from pathlib import Path
import sqlite3
import struct
import numpy as np

HEADER=struct.Struct('<BI')
KINDS={0:'identity',1:'dense',2:'lowrank',3:'shared'}


def pack(kind,arrays=(),rank=0):
    return HEADER.pack(kind,rank)+b''.join(np.asarray(a,dtype='<f8').tobytes() for a in arrays)


def execute(blob,x,d,basis=None):
    kind,r=HEADER.unpack_from(blob)
    a=np.frombuffer(blob,dtype='<f8',offset=HEADER.size)
    if kind==0:return x.copy()
    if kind==1:return x@a.reshape(d,d).T
    if kind==2:
        u=a[:d*r].reshape(d,r);v=a[d*r:].reshape(r,d)
        return x+(x@v.T)@u.T
    if kind==3:
        if basis is None:raise ValueError('missing shared basis')
        # Do not materialize the full operator matrix.
        return x+np.einsum('r,rnd->nd',a,np.einsum('ne,rde->rnd',x,basis))
    raise ValueError('invalid kind')


def error(blob,x,y,d,basis=None):
    return float(np.linalg.norm(execute(blob,x,d,basis)-y)/max(np.linalg.norm(y),1e-12))


def independent(t,xv,yv,tol):
    d=len(t)
    candidates=[pack(1,[t])]
    identity=pack(0)
    if error(identity,xv,yv,d)<=tol:candidates.append(identity)
    u,s,v=np.linalg.svd(t-np.eye(d),full_matrices=False)
    for r in range(1,d//2):
        b=pack(2,[u[:,:r]*s[:r],v[:r]],r)
        if error(b,xv,yv,d)<=tol:
            candidates.append(b);break
    return min(candidates,key=len)


def truth(op,d,seed):
    rng=np.random.default_rng(seed)
    # Same four latent outer products reused across the structured third.
    u=rng.normal(size=(d,4))/np.sqrt(d)
    v=rng.normal(size=(4,d))/np.sqrt(d)
    rng=np.random.default_rng(seed+1000+op)
    t=np.eye(d)
    if op%3==0:
        c=rng.normal(size=4)*3
        t+=(u*c)@v
    elif op%3==1:
        a,b=rng.choice(d,2,replace=False)
        t[[a,b]]=t[[b,a]]
    else:t=t[rng.permutation(d)]
    return t


def connect(path,write=False):
    c=sqlite3.connect(path if write else f'file:{path}?mode=ro',uri=not write)
    c.execute('PRAGMA cache_size=-256')
    c.execute('PRAGMA mmap_size=0')
    if write:
        c.execute('CREATE TABLE operators (id INTEGER PRIMARY KEY,payload BLOB NOT NULL)')
    else:c.execute('PRAGMA query_only=ON')
    return c


def build(root,count=192,d=32,seed=40,tol=1e-4,dictionary_rank=4,group_dictionary=False,noise=0.0,group_tolerance=1e-6):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    # This training stage holds learned matrices. Runtime does not hold this array.
    matrices=[]
    for op in range(count):
        rng=np.random.default_rng(seed+2000+op)
        x=rng.normal(size=(2*d,d))
        y=x@truth(op,d,seed).T
        if noise:y+=rng.normal(size=y.shape)*noise
        matrices.append(np.linalg.lstsq(x,y,rcond=None)[0].T)
    matrices=np.array(matrices)
    residuals=(matrices-np.eye(d)).reshape(count,-1)
    # Uncentered SVD dictionary learned from training estimates, no family labels.
    _,_,vt=np.linalg.svd(residuals,full_matrices=False)
    basis=vt[:dictionary_rank].reshape(dictionary_rank,d,d)
    coeff=residuals@vt[:dictionary_rank].T
    # Group by row/column subspaces estimated from training-only residuals.
    # No generator family labels and no test vectors enter grouping.
    groups=[]
    if group_dictionary:
        for op,t in enumerate(matrices):
            u,sv,v=np.linalg.svd(t-np.eye(d),full_matrices=False)
            r=int(np.sum(sv>max(1e-8,sv[0]*1e-4)))
            if not 0<r<=dictionary_rank:continue
            left=u[:,:r]@u[:,:r].T;right=v[:r].T@v[:r]
            matched=False
            for g in groups:
                if r==g['r'] and max(np.linalg.norm(left-g['left']),np.linalg.norm(right-g['right']))<=group_tolerance:
                    g['ids'].append(op);matched=True;break
            if not matched:groups.append(dict(r=r,left=left,right=right,ids=[op]))
        best_saving=0;best_basis=np.zeros((0,d,d))
        # Evaluate common-basis cost + coefficients against compact independent
        # alternatives, using the validation set. Never use held-out test scores.
        for g in groups:
            if len(g['ids'])<2:continue
            _,_,gv=np.linalg.svd(residuals[g['ids']],full_matrices=False)
            gb=gv[:dictionary_rank].reshape(-1,d,d)
            savings=-gb.nbytes
            for op in g['ids']:
                rng=np.random.default_rng(seed+3000+op)
                xv=rng.normal(size=(3*d,d));yv=xv@truth(op,d,seed).T
                compact=independent(matrices[op],xv,yv,tol)
                code=residuals[op]@gb.reshape(len(gb),-1).T
                candidate=pack(3,[code],len(gb))
                if error(candidate,xv,yv,d,gb)<=tol:
                    savings+=max(0,len(compact)-len(candidate))
            if savings>best_saving:best_saving=savings;best_basis=gb
        basis=best_basis
        coeff=residuals@basis.reshape(len(basis),d*d).T
    con={}
    for method in ('dense','adaptive','shared'):
        path=root/(method+'.sqlite')
        if path.exists():path.unlink()
        con[method]=connect(path,True)
    counts={m:{v:0 for v in KINDS.values()} for m in con}
    payloads={m:0 for m in con}
    validation_errors={m:[] for m in con}
    for op,t in enumerate(matrices):
        rng=np.random.default_rng(seed+3000+op)
        xv=rng.normal(size=(3*d,d));yv=xv@truth(op,d,seed).T
        dense=pack(1,[t]);compact=independent(t,xv,yv,tol)
        b=pack(3,[coeff[op]],len(basis))
        # A shared candidate must be both smaller per item and validation-accurate.
        selected=b if len(basis)>0 and len(b)<len(compact) and error(b,xv,yv,d,basis)<=tol else compact
        for method,blob in [('dense',dense),('adaptive',compact),('shared',selected)]:
            con[method].execute('INSERT INTO operators VALUES (?,?)',(op,blob))
            counts[method][KINDS[HEADER.unpack_from(blob)[0]]]+=1
            payloads[method]+=len(blob)
            validation_errors[method].append(error(blob,xv,yv,d,basis))
    # If the entire dictionary costs more than it saves, disable it globally.
    use_shared=payloads['shared']+basis.nbytes < payloads['adaptive']
    if not use_shared:
        con['shared'].execute('DELETE FROM operators')
        for op,blob in con['adaptive'].execute('SELECT id,payload FROM operators'):
            con['shared'].execute('INSERT INTO operators VALUES (?,?)',(op,blob))
        counts['shared']=dict(counts['adaptive']);payloads['shared']=payloads['adaptive']
        validation_errors['shared']=list(validation_errors['adaptive'])
        basis=np.zeros((0,d,d))
    np.save(root/'basis.npy',basis,allow_pickle=False)
    for c in con.values():c.commit();c.close()
    meta=dict(seed=seed,count=count,d=d,tolerance=tol,requested_dictionary_rank=dictionary_rank,
              shared_basis_payload_bytes=basis.nbytes,shared_basis_file_bytes=(root/'basis.npy').stat().st_size,
              shared_enabled=use_shared,dictionary_strategy="grouped" if group_dictionary else "global",
              training_output_noise_std=noise,group_tolerance=group_tolerance,training_groups=len(groups),
              counts=counts,payload_bytes=payloads,
              validation_error_max={m:max(v) for m,v in validation_errors.items()},
              database_file_bytes={m:(root/(m+'.sqlite')).stat().st_size for m in con},
              note='Gaussian fit/validation samples separate; matrix dictionary from training only. No linguistic routing.')
    (root/'metadata.json').write_text(json.dumps(meta,indent=2))
    return meta


class Paged:
    def __init__(self,root,method,budget):
        root=Path(root)
        self.meta=json.loads((root/'metadata.json').read_text())
        self.d=self.meta['d'];self.method=method
        self.basis=np.load(root/'basis.npy',allow_pickle=False) if method=='shared' else None
        fixed=0 if self.basis is None else self.basis.nbytes
        if budget<fixed:raise ValueError('budget smaller than shared resident basis')
        self.fixed=fixed;self.capacity=budget-fixed
        self.cache=OrderedDict();self.used=0;self.peak=fixed
        self.hits=0;self.misses=0;self.logical_bytes=0
        self.conn=connect(root/(method+'.sqlite'))

    def get(self,op):
        if op in self.cache:
            self.hits+=1;self.cache.move_to_end(op);return self.cache[op]
        self.misses+=1
        row=self.conn.execute('SELECT payload FROM operators WHERE id=?',(int(op),)).fetchone()
        if row is None:raise KeyError(op)
        b=row[0];self.logical_bytes+=len(b)
        if len(b)<=self.capacity:
            while self.used+len(b)>self.capacity:
                _,old=self.cache.popitem(last=False);self.used-=len(old)
            self.cache[op]=b;self.used+=len(b)
            self.peak=max(self.peak,self.fixed+self.used)
        return b

    def apply(self,x,op):
        return execute(self.get(op),x,self.d,self.basis)

    def close(self):self.conn.close()
