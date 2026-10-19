"""Label-free probe generation; no target program or test outputs inspected."""
import random
from core import key


def generate(kind,seed,count=64):
    rng=random.Random(seed);values={}
    while len(values)<count:
        if kind=='int':x=rng.randint(-200,200)
        elif kind=='text':x=''.join(rng.choice('aBxY ,_') for _ in range(rng.randint(0,18)))
        elif kind=='list':x=[rng.randint(-12,12) for _ in range(rng.randint(0,8))]
        else:raise ValueError('unknown domain')
        values[key(x)]=x
    return list(values.values())
