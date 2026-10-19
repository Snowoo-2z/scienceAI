"""Inspect KAIROS's pinned tokenizer; no checkpoint retraining or modification.
A lossy tokenizer collision is an input-information loss, not an LM score.
"""
import json
from pathlib import Path
import subprocess

COMMIT='b07d2dcf6b781ca463675fb5de6be8dc02c0c5f3'


def read(path):return subprocess.check_output(['git','show',f'{COMMIT}:{path}'],text=True)


def run():
    # Reviewed local peer source at an immutable commit. Its CLI main is not run.
    scope={'__name__':'silex_pinned_tokenizer_review','__file__':str(Path.cwd()/'.cache/silex-peer/tinygpt.py')}
    exec(compile(read('demo/tinygpt.py'),f'{COMMIT}:demo/tinygpt.py','exec'),scope)
    metadata=json.loads(read('models/tinygpt.npz.json'))
    tok=scope['Tokenizer'].from_state(metadata['tokenizer'])
    pairs=[('mon code est zorpulix.','mon code est vaxotron.'),
           ('BONJOUR   MONDE','bonjour monde'),
           ('Je vis à Erquy.','Je vis à Plouescat.')]
    result=[]
    for a,b in pairs:
        x,y=tok.encode(a),tok.encode(b)
        result.append(dict(first=a,second=b,first_ids=x,second_ids=y,identical=x==y,
                           first_unknowns=x.count(tok.stoi['<unk>']),second_unknowns=y.count(tok.stoi['<unk>']),
                           decoded_first=tok.decode(x),decoded_second=tok.decode(y)))
    assert result[0]['identical'] and result[0]['first_unknowns']>0
    assert result[1]['identical']
    return dict(peer_commit=COMMIT,mode=tok.mode,vocabulary=len(tok.itos),examples=result,
                inference='Identical encoded contexts necessarily give identical deterministic model logits; checkpoint forward not executed here.',
                bpc_caveat='Word-token loss divided by source-character count is not lossless character coding cost: unknown spellings, case and whitespace are omitted.',
                split_caveat='train() builds tokenizer on entire text before splitting train/validation IDs; vocabulary frequencies include validation text.')

if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
