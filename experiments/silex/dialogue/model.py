"""Tiny conditional recurrent generator. No Transformer, lookup, RAG or external API.
Question encoding is a bag of words: intentionally weak order-blind control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def tokens(text):
    return re.findall(r"[\w]+|[^\w\s]", text.lower(), flags=re.UNICODE)


def detokenize(words):
    text = ' '.join(words)
    text = re.sub(r"\s*'\s*", "'", text)
    return re.sub(r'\s+([.,])', r'\1', text)


def vocabulary(pairs):
    return ['<pad>', '<bos>', '<eos>', '<unk>'] + sorted({w for q, a in pairs for w in tokens(q + ' ' + a)})


def encode_questions(questions, vocab):
    ids = {s: i for i, s in enumerate(vocab)}
    bag = np.zeros((len(questions), len(vocab)), np.float32)
    for i, q in enumerate(questions):
        words = tokens(q)
        for w in words:
            bag[i, ids.get(w, 3)] += 1 / max(len(words), 1)
    return bag


def init(v, h=64, seed=42):
    rng = np.random.default_rng(seed)
    def normal(shape, scale):
        return (rng.normal(size=shape) * scale).astype(np.float32)
    return dict(Q=normal((v,h),.1), E=normal((v,h),.1),
                R=normal((h,h),1/np.sqrt(h)), O=normal((h,v),.1),
                b=np.zeros(h,np.float32), c=np.zeros(v,np.float32))


def loss_grad(p, bag, inputs, targets):
    b, length = inputs.shape
    h = np.zeros((b, len(p['b'])), dtype=p['b'].dtype)
    context = bag @ p['Q']
    count = max(int((targets != 0).sum()), 1)
    cache = []
    loss = 0.
    for t in range(length):
        old = h
        h = np.tanh(old @ p['R'] + p['E'][inputs[:,t]] + context + p['b'])
        logits = h @ p['O'] + p['c']
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        mask = targets[:,t] != 0
        loss -= np.log(probs[np.arange(b),targets[:,t]] + 1e-30)[mask].sum() / count
        dy = probs.copy()
        dy[np.arange(b),targets[:,t]] -= 1
        dy *= mask[:,None] / count
        cache.append((old, h, dy))
    g = {k:np.zeros_like(v) for k,v in p.items()}
    dh = np.zeros_like(h)
    for t in reversed(range(length)):
        old, h, dy = cache[t]
        g['O'] += h.T @ dy
        g['c'] += dy.sum(axis=0)
        dz = (dy @ p['O'].T + dh) * (1 - h*h)
        g['R'] += old.T @ dz
        np.add.at(g['E'], inputs[:,t], dz)
        g['Q'] += bag.T @ dz
        g['b'] += dz.sum(axis=0)
        dh = dz @ p['R'].T
    return float(loss), g


def generate(p, vocab, question, limit=48):
    bag = encode_questions([question], vocab)
    context = bag @ p['Q']
    h = np.zeros((1,len(p['b'])), np.float32)
    word = 1
    output = []
    stopped = False
    for _ in range(limit):
        h = np.tanh(h @ p['R'] + p['E'][word] + context + p['b'])
        logits = (h @ p['O'] + p['c'])[0]
        # Exclude control symbols that are not valid generated text.
        logits[[0,1,3]] = -np.inf
        word = int(np.argmax(logits))
        if word == 2:
            stopped = True
            break
        output.append(vocab[word])
    known = set(vocab)
    unknown = [w for w in tokens(question) if w not in known]
    return dict(answer=detokenize(output), unknown_words=unknown,
                generated_tokens=len(output), stopped_on_eos=stopped)


def load():
    meta = json.loads((ROOT/'metadata.json').read_text())
    with np.load(ROOT/'weights.npz', allow_pickle=False) as data:
        p = {k:data[k] for k in ('Q','E','R','O','b','c')}
    return p, meta


def check_gradients():
    p = {k:v.astype(np.float64) for k,v in init(7,4).items()}
    bag = np.array([[0,0,0,0,1,0,0],[0,0,0,0,0,.5,.5]],np.float64)
    x=np.array([[1,4,5],[1,6,0]])
    y=np.array([[4,5,2],[6,2,0]])
    loss,g=loss_grad(p,bag,x,y)
    rng=np.random.default_rng(1)
    errors=[]
    for k in p:
        for _ in range(5):
            index=tuple(int(rng.integers(n)) for n in p[k].shape)
            original=p[k][index]
            p[k][index]=original+1e-5
            a,_=loss_grad(p,bag,x,y)
            p[k][index]=original-1e-5
            b,_=loss_grad(p,bag,x,y)
            p[k][index]=original
            error=abs((a-b)/2e-5-g[k][index])
            errors.append(error)
            assert error < 1e-7,(k,index,error)
    print('30 finite-difference gradient checks PASS; max error',max(errors))


def train(epochs):
    corpus=json.loads((ROOT/'corpus.json').read_text())
    pairs=corpus['train']
    vocab=vocabulary(pairs)
    ids={w:i for i,w in enumerate(vocab)}
    bag=encode_questions([q for q,a in pairs],vocab)
    answers=[[ids[w] for w in tokens(a)]+[2] for q,a in pairs]
    targets=np.zeros((len(pairs),max(map(len,answers))),np.int64)
    inputs=np.zeros_like(targets)
    for i,a in enumerate(answers):
        targets[i,:len(a)]=a
        inputs[i,:len(a)]=[1]+a[:-1]
    p=init(len(vocab))
    m={k:np.zeros_like(v) for k,v in p.items()}
    v={k:np.zeros_like(v) for k,v in p.items()}
    history=[]
    start=time.perf_counter()
    for step in range(1,epochs+1):
        loss,g=loss_grad(p,bag,inputs,targets)
        norm=np.sqrt(sum(float((a*a).sum()) for a in g.values()))
        for k in p:
            grad=g[k]*min(1.,1./max(norm,1e-12))
            m[k]=.9*m[k]+.1*grad
            v[k]=.999*v[k]+.001*grad*grad
            p[k]-=.01*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        if step==1 or step%100==0 or step==epochs:
            history.append(dict(step=step,loss_before_update=loss))
            print(step,round(loss,6),flush=True)
    np.savez(ROOT/'weights.npz',**p)
    meta=dict(model='Silex-RNN-laboratoire-v0', architecture='bag_question_plus_shared_tanh_recurrence',
              vocab=vocab, hidden=64, seed=42, epochs=epochs, training_pairs=len(pairs),
              parameter_count=sum(a.size for a in p.values()), parameter_bytes=sum(a.nbytes for a in p.values()),
              training_seconds=time.perf_counter()-start,
              training_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              corpus_sha256=hashlib.sha256((ROOT/'corpus.json').read_bytes()).hexdigest(),
              weights_sha256=hashlib.sha256((ROOT/'weights.npz').read_bytes()).hexdigest(),
              history=history, numpy=np.__version__, date='2026-10-19',
              warning='Synthetic tiny corpus. Not general intelligence; no factual reliability. No learned plastic-memory integration.')
    (ROOT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')


def evaluate():
    p,meta=load()
    corpus=json.loads((ROOT/'corpus.json').read_text())
    results={}
    for split in ('train','paraphrases'):
        rows=[]
        for q,a in corpus[split]:
            row=dict(question=q,expected=a,**generate(p,meta['vocab'],q))
            row['exact_match']=tokens(row['answer'])==tokens(a)
            rows.append(row)
        results[split]=dict(exact_match=sum(r['exact_match'] for r in rows)/len(rows), examples=rows)
    results['unseen']=[dict(question=q,**generate(p,meta['vocab'],q)) for q in corpus['unseen']]
    results['inference_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    (ROOT/'responses.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','eval','test'])
    parser.add_argument('--epochs',type=int,default=800)
    args=parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    if args.action=='train': train(args.epochs)
    elif args.action=='eval': evaluate()
    else: check_gradients()
