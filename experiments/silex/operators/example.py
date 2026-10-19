"""Readable outputs from learned operators; no language-model claim.
The target transitions are used ONLY to generate training examples and to check
answers. The predicted output uses learned factors, not the target matrices.
"""
import json
import numpy as np
from experiment import targets, fit_shared, apply


def main():
    seed=20
    truth=targets('swaps',16,8,seed+200)
    rng=np.random.default_rng(seed+100)
    x=rng.normal(size=(8,32,16))
    y=np.einsum('knd,ked->kne',x,truth)
    p,_=fit_shared(x,y,8,seed,2400,True)
    records=[]
    for text,program in [('ABCDEFGHIJKLMNOP',[0,1,0]),
                         ('bonjour_les_amis',[2,3,2,0,1]),
                         ('0123456789abcdef',[0,1,2,3,4,5,6,7]*8)]:
        assert len(text)==16
        initial=np.array([[ord(c) for c in text]],dtype=np.float64)
        pred=initial.copy()
        expected=initial.copy()
        for op in program:
            pred=apply(('shared',p),pred,op)
            expected=expected@truth[op].T
        # Round for discrete symbol rendering only. Numeric benchmark uses no rounding.
        output=''.join(chr(int(v)) for v in np.rint(pred[0]))
        target=''.join(chr(int(v)) for v in expected[0])
        records.append(dict(input=text,operator_ids=program,output=output,expected=target,
                            matches=output==target,max_unrounded_absolute_error=float(abs(pred-expected).max())))
    print(json.dumps(dict(description='Symbol permutations from learned atomic operators. Not conversation.',
                          trained_on_compositions=False,examples=records),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
