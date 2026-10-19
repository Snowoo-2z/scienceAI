"""Fresh-process warm-cache S5 benchmark, not an LLM throughput claim."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time
import numpy as np
from library import Paged,build,truth


def proc_io():
    try:return {k:int(v) for k,v in (s.split(':') for s in Path('/proc/self/io').read_text().splitlines())}
    except OSError:return {}


def rss():
    try:
        return int(next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')).split()[1])*1024
    except (OSError,StopIteration):return None


def cgroup():
    try:
        rel=next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        return int((Path('/sys/fs/cgroup')/rel.lstrip('/')/'memory.current').read_text())
    except (OSError,StopIteration):return None


def bench(root,method,budget,workload,requests,seed,cold_file=False):
    advice=False
    if cold_file and hasattr(os,'posix_fadvise'):
        with open(Path(root)/(method+'.sqlite'),'rb') as f:
            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        advice=True
    obj=Paged(root,method,budget)
    rng=np.random.default_rng(seed)
    count=obj.meta['count'];d=obj.d
    if workload=='uniform':ids=rng.integers(count,size=requests)
    else:
        p=1/np.arange(1,count+1,dtype=float)**1.2;p/=p.sum()
        ids=rng.choice(count,size=requests,p=p)
    # No test outputs are used for representation choice. Samples seeded independently.
    x=rng.normal(size=(1,d));times=[];checksum=0.
    before=proc_io();cg0=cgroup();start=time.perf_counter()
    for op in ids:
        t=time.perf_counter();y=obj.apply(x,int(op));times.append(time.perf_counter()-t)
        checksum+=float(y.sum())
    elapsed=time.perf_counter()-start;after=proc_io();cg1=cgroup()
    measured=dict(hits=obj.hits,misses=obj.misses,logical_payload_read_bytes=obj.logical_bytes,
                  peak_cached_and_basis_payload_bytes=obj.peak)
    # Quality pass after timed pass. Does not affect cache metrics above.
    err=[]
    for op in range(count):
        xt=rng.normal(size=(7,d));expected=xt@truth(op,d,obj.meta['seed']).T
        actual=obj.apply(xt,op)
        err.append(float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12)))
    obj.close()
    return dict(method=method,workload=workload,requests=requests,request_seed=seed,
                budget_payload_bytes=budget,**measured,seconds=elapsed,
                p50_ms=float(np.percentile(times,50)*1000),p95_ms=float(np.percentile(times,95)*1000),
                checksum=checksum,test_error_max=max(err),test_error_mean=float(np.mean(err)),
                test_pass_fraction=float(np.mean(np.array(err)<=obj.meta['tolerance'])),
                physical_read_bytes_delta=after.get('read_bytes',0)-before.get('read_bytes',0) if before and after else None,
                rss_current_bytes=rss(),rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cgroup_memory_current_before=cg0,cgroup_memory_current_after=cg1,cgroup_memory_current_after_quality=cgroup(),
                cgroup_warning='May include other processes, page cache and running preview; not attributable to this model.',
                cache_regime='own DB file DONTNEED advisory before open; not a controlled cold device' if advice else 'warm OS cache, initially empty application LRU; no cache flush',
                cold_file_advice_issued=advice,
                backend='numpy_'+np.__version__+'_sqlite3',python=platform.python_version(),hardware=platform.machine(),
                model='synthetic_learned_linear_operator_library',quantization='none_float64',
                context='one vector and provided operator ID',date='2026-10-19',energy=None,language_quality=None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','bench'])
    p.add_argument('--cold-file',action='store_true')
    p.add_argument('--group-dictionary',action='store_true')
    p.add_argument('--noise',type=float,default=0.)
    p.add_argument('--group-tolerance',type=float,default=1e-6)
    p.add_argument('--root',default='.cache/silex-s5')
    p.add_argument('--count',type=int,default=192)
    p.add_argument('--d',type=int,default=32)
    p.add_argument('--seed',type=int,default=40)
    p.add_argument('--method',choices=['dense','adaptive','shared'],default='adaptive')
    p.add_argument('--budget',type=int,default=65536)
    p.add_argument('--workload',choices=['uniform','zipf'],default='uniform')
    p.add_argument('--requests',type=int,default=3000)
    a=p.parse_args()
    if a.count<2 or a.d<4 or a.requests<1 or a.budget<0 or a.noise<0 or a.group_tolerance<=0:p.error('invalid size/budget')
    result=build(a.root,a.count,a.d,a.seed,group_dictionary=a.group_dictionary,noise=a.noise,group_tolerance=a.group_tolerance) if a.action=='build' else bench(a.root,a.method,a.budget,a.workload,a.requests,a.seed+4000,a.cold_file)
    print(json.dumps(result,indent=2))
