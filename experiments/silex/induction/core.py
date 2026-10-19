"""Bounded programming-by-example, with disk-backed hypothesis sets.
Handwritten primitives; discovered compositions; no free-language parser.
All execution uses a closed allowlist, never eval or arbitrary Python code.
"""
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3

VERSION='silex-induction-v1'
INVALID=object()


def encode(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def bounded(value):
    if type(value) is int:return abs(value)<=10**9
    if type(value) is str:return len(value)<=256
    if type(value) is list:
        return len(value)<=32 and all(type(v) in (int,str) and bounded(v) for v in value)
    return False


def ints(x):
    if not all(type(v) is int for v in x):raise TypeError('integer list required')
    return x


def strings(x):
    if not all(type(v) is str for v in x):raise TypeError('string list required')
    return x


def unique(x):return list(dict.fromkeys(x))

# Ordered, explicit interpreter vocabulary. Constants are supplied, not inferred.
OPS={
 'strip':((str,),lambda x:x.strip()),
 'lower':((str,),lambda x:x.lower()),
 'upper':((str,),lambda x:x.upper()),
 'split_space':((str,),lambda x:x.split()),
 'split_comma':((str,),lambda x:x.split(',')),
 'underscore_to_space':((str,),lambda x:x.replace('_',' ')),
 'remove_spaces':((str,),lambda x:x.replace(' ','')),
 'reverse':((str,list),lambda x:x[::-1]),
 'take2':((str,list),lambda x:x[:2]),
 'drop1':((str,list),lambda x:x[1:]),
 'sort':((list,),lambda x:sorted(x)),
 'unique':((list,),unique),
 'length':((str,list),len),
 'join_space':((list,),lambda x:' '.join(strings(x))),
 'sum':((list,),lambda x:sum(ints(x))),
 'minimum':((list,),lambda x:min(ints(x))),
 'maximum':((list,),lambda x:max(ints(x))),
 'abs_each':((list,),lambda x:[abs(v) for v in ints(x)]),
 'double_each':((list,),lambda x:[2*v for v in ints(x)]),
 'add1_each':((list,),lambda x:[v+1 for v in ints(x)]),
 'positive_only':((list,),lambda x:[v for v in ints(x) if v>0]),
 'add1':((int,),lambda x:x+1),
 'add2':((int,),lambda x:x+2),
 'sub1':((int,),lambda x:x-1),
 'mul2':((int,),lambda x:x*2),
 'mul3':((int,),lambda x:x*3),
 'negate':((int,),lambda x:-x),
 'absolute':((int,),abs),
 'square':((int,),lambda x:x*x),
}


def step(name,value):
    if name not in OPS or not bounded(value):return INVALID
    types,fn=OPS[name]
    if type(value) not in types:return INVALID
    try:out=fn(value)
    except (ValueError,TypeError,OverflowError):return INVALID
    return out if bounded(out) else INVALID


def execute(program,value):
    if not bounded(value) or len(program)>8:return INVALID
    for name in program:
        value=step(name,value)
        if value is INVALID:break
    return value


def key(value):return '!invalid' if value is INVALID else encode(value)


class Space:
    """SQLite stores every matching syntax tree, NOT only training behaviors.
    Collapsing hypotheses identical on demonstrations would erase ambiguity.
    """
    def __init__(self,path):
        self.path=Path(path)
        self.db=sqlite3.connect(self.path)
        self.db.execute('PRAGMA cache_size=-256')
        self.db.execute('PRAGMA mmap_size=0')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('CREATE TABLE IF NOT EXISTS hypotheses (id INTEGER PRIMARY KEY, depth INTEGER, program TEXT UNIQUE)')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)')
        version=self.get('version')
        if version is not None and version!=VERSION:raise ValueError('unsupported interpreter version')

    def close(self):self.db.close()
    def get(self,name):
        row=self.db.execute('SELECT value FROM metadata WHERE key=?',(name,)).fetchone()
        return None if row is None else json.loads(row[0])
    def put(self,name,value):self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(name,encode(value)))
    def count(self):return self.db.execute('SELECT count(*) FROM hypotheses').fetchone()[0]
    def programs(self):
        for ident,program in self.db.execute('SELECT id,program FROM hypotheses ORDER BY id'):
            yield ident,json.loads(program)

    def fit(self,examples,depth=3,max_nodes=100000,max_matches=10000):
        if not 0<=depth<=6 or max_nodes<1 or max_matches<1:raise ValueError('invalid search budget')
        if not 1<=len(examples)<=16:raise ValueError('supply 1..16 examples')
        if any(set(e)!={'input','output'} or not bounded(e['input']) or not bounded(e['output']) for e in examples):
            raise ValueError('expected bounded JSON integer/string/flat-list pairs')
        self.db.execute('DELETE FROM hypotheses');self.db.execute('DELETE FROM metadata')
        expected=[key(e['output']) for e in examples]
        stats=dict(nodes=0,matches=0,complete=True,depth=depth,max_nodes=max_nodes,max_matches=max_matches)
        def visit(program,values):
            if stats['nodes']>=max_nodes:
                stats['complete']=False;return False
            stats['nodes']+=1
            if [key(v) for v in values]==expected:
                if stats['matches']>=max_matches:
                    stats['complete']=False;return False
                self.db.execute('INSERT INTO hypotheses(depth,program) VALUES (?,?)',(len(program),encode(program)))
                stats['matches']+=1
            if len(program)==depth:return True
            for name in OPS:
                next_values=[step(name,v) for v in values]
                if any(v is INVALID for v in next_values):continue
                if not visit(program+[name],next_values):return False
            return True
        visit([],[e['input'] for e in examples])
        self.put('version',VERSION);self.put('examples',examples);self.put('search',stats)
        self.db.commit()
        return stats

    def shortest(self,value):
        row=self.db.execute('SELECT program FROM hypotheses ORDER BY depth,id LIMIT 1').fetchone()
        if row is None:return dict(status='no_hypothesis')
        p=json.loads(row[0]);out=execute(p,value)
        return dict(status='invalid' if out is INVALID else 'answer',output=None if out is INVALID else out,program=p)

    def predict(self,value):
        if not bounded(value):raise ValueError('input outside resource limits')
        n=self.count()
        if not n:return dict(status='no_hypothesis',hypotheses=0)
        search=self.get('search')
        if not search or not search['complete']:
            return dict(status='incomplete_search',hypotheses=n)
        output=INVALID;first=True
        for _,p in self.programs():
            out=execute(p,value)
            if first:output=out;first=False
            elif key(out)!=key(output):return dict(status='ambiguous',hypotheses=n)
        if output is INVALID:return dict(status='unsupported',hypotheses=n)
        return dict(status='consensus',output=output,hypotheses=n,
                    guarantee='agreement_only_within_enumerated_DSL_and_depth')

    def clarify(self,pool,max_partition_bytes=1048576):
        if len(pool)>64 or any(not bounded(x) for x in pool):raise ValueError('invalid query pool')
        seen={key(e['input']) for e in (self.get('examples') or [])}
        pool=list({key(x):x for x in pool if key(x) not in seen}.values())
        n=self.count()
        if not n:return dict(status='no_hypothesis')
        best=None
        # One input's partition at a time: bounded workspace instead of 64 maps.
        # Estimate is not an RSS guarantee; benchmarks also impose RLIMIT_AS.
        for x in pool:
            c=Counter();estimated_bytes=0
            for _,p in self.programs():
                out=key(execute(p,x))
                if out not in c:
                    estimated_bytes+=len(out.encode('utf-8'))+128
                    if estimated_bytes>max_partition_bytes:
                        return dict(status='query_budget_exhausted',hypotheses=n)
                c[out]+=1
            if len(c)<2:continue
            entropy=-sum((v/n)*math.log2(v/n) for v in c.values())
            if best is None or entropy>best['entropy_bits']:
                best=dict(status='question',input=x,entropy_bits=entropy,partitions=len(c),
                          largest_partition=max(c.values()),hypotheses=n)
        return best or dict(status='no_disagreement_in_pool',hypotheses=n)

    def observe(self,value,output):
        if not bounded(value) or not bounded(output):raise ValueError('invalid observation')
        examples=self.get('examples') or []
        if len(examples)>=16:raise ValueError('example limit reached')
        # Streaming cursor on separate SELECT; DELETEs committed only at end.
        for ident,p in self.programs():
            if key(execute(p,value))!=key(output):
                self.db.execute('DELETE FROM hypotheses WHERE id=?',(ident,))
        examples.append(dict(input=value,output=output));self.put('examples',examples);self.db.commit()
        return self.count()

    def revise_if_empty(self,max_depth=4,max_nodes=100000,max_matches=10000):
        """Widen only after contradiction/no fit, using observations, not a target.
        Does not fix misspecification which has not yet produced a contradiction.
        """
        search=self.get('search')
        if not search:raise ValueError('fit first')
        if self.count():return dict(status='unchanged',hypotheses=self.count(),search=search)
        examples=self.get('examples')
        for depth in range(search['depth']+1,max_depth+1):
            search=self.fit(examples,depth,max_nodes,max_matches)
            if self.count() or not search['complete']:break
        return dict(status='revised' if self.count() else 'no_hypothesis',
                    hypotheses=self.count(),search=search)

    def copy_to(self,path):
        dest=sqlite3.connect(path);self.db.backup(dest);dest.close()
