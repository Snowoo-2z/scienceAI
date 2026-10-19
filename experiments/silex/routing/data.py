"""Authored controlled-French task, not a general language corpus.
All splits/templates fixed before running the two router ablations.
The supplied program boundary (one instruction per step) is not learned.
"""
WORDS=['un','deux','trois','quatre','cinq','six','sept','huit','neuf']
TRAIN=[
    'échange les positions {a} et {b}',
    'inverse les éléments {a} et {b}',
    'permute les cases {a} et {b}',
    'intervertis les positions {a} et {b}',
    'fais un échange entre {a} et {b}',
]
VALID=[
    'peux-tu échanger les positions {a} et {b} ?',
    'merci de permuter les éléments {a} et {b}',
]
TEST=[
    'dans la liste, intervertis les cases {a} et {b}, sans toucher aux autres',
    'je veux que les positions {b} et {a} soient échangées',
    'échange seulement les deux éléments aux positions {a} et {b}',
    'inverse les places {b} et {a} s’il te plaît',
]
NEG_TRAIN=["n'échange pas les positions {a} et {b}",
           'ne permute pas les cases {a} et {b}']
NEG_VALID=['ne touche pas aux positions {a} et {b}',
           'échange les positions {a} et 16']
NEG_TEST=['il ne faut surtout pas échanger les positions {a} et {b}',
          'garde les éléments {a} et {b} à leur place',
          'supprime les positions {a} et {b}',
          'additionne les éléments {a} et {b}']
GENERAL={
 'train':['bonjour','raconte une histoire','quelle heure est-il','trie toute la liste',
          'déplace tous les éléments','donne une recette','inverse toute la liste','efface tout'],
 'validation':['salut à toi','écris un poème','combien font deux plus trois','tourne la liste'],
 'test':['WOW','quelle est la capitale du japon ?', 'explique la gravité',
         'remplace le premier élément par zéro','échange les positions un et neuf',
         'échange les positions trois et onze','la liste est déjà correcte',
         'ne fais absolument rien']}


def expand(templates,negative=False):
    rows=[]
    for op in range(8):
        for spelling in ('digits','words'):
            a,b=(str(op+1),str(op+2)) if spelling=='digits' else (WORDS[op],WORDS[op+1])
            for template in templates:
                rows.append(dict(text=template.format(a=a,b=b),label=-1 if negative else op,
                                 spelling=spelling,kind='unsupported' if negative else 'supported'))
    return rows


def dataset():
    result={}
    for name,templates,neg in [('train',TRAIN,NEG_TRAIN),('validation',VALID,NEG_VALID),('test',TEST,NEG_TEST)]:
        result[name]=expand(templates)+expand(neg,True)+[
            dict(text=q,label=-1,kind='unsupported',spelling=None) for q in GENERAL[name]]
    return result
