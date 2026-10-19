"""Reproduce KAIROS pinned router and compare substring vs exact scoring.
Does not alter a colleague's branch or tune their router.
Run only the reviewed pinned stdlib-only source, not arbitrary harvested code.
"""
import decimal
import json
import subprocess
import sys
import types

COMMIT='ff4b346c3df8d1a90bea978f44a3dd9d721dcd2d'


def source(path):return subprocess.check_output(['git','show',f'{COMMIT}:{path}'],text=True)


def exact(a,b):
    a=a.strip().casefold().replace(',','.');b=b.strip().casefold().replace(',','.')
    try:
        x,y=decimal.Decimal(a),decimal.Decimal(b)
        return x.is_finite() and y.is_finite() and x==y
    except decimal.InvalidOperation:return a==b


def main():
    mod=types.ModuleType('silex_reviewed_harvest');sys.modules[mod.__name__]=mod
    exec(compile(source('demo/harvest.py'),f'{COMMIT}:demo/harvest.py','exec'),mod.__dict__)
    cards,stats=mod.harvest();items=json.loads(source('demo/eval_questions.json'))
    answered=[];missing=0
    for row in items:
        value,fn,proof=mod.route_and_call(cards,row['q'])
        if value is None:missing+=1;continue
        eligible=row['kind']=='competence' and bool(row['expect'])
        old=eligible and row['expect'].lower().replace(',','.') in value.lower().replace(',','.')
        good=eligible and exact(value,row['expect'])
        answered.append(dict(**row,result=value,function=fn,proof=proof,
                             substring_correct=bool(old),exact_correct=bool(good)))
    return dict(peer_commit=COMMIT,questions=len(items),unanswered=missing,answered=len(answered),
                substring_correct=sum(r['substring_correct'] for r in answered),
                exact_correct=sum(r['exact_correct'] for r in answered),details=answered,
                caveat='Exact decimal/string comparison audited for the five returned answers; not a universal semantic evaluator.')


if __name__=='__main__':
    assert not exact('31','1') and exact('343.0','343') and not exact('nan','NaN')
    print(json.dumps(main(),ensure_ascii=False,indent=2))
