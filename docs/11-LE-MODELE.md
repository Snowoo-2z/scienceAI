# 11 — Le vrai modèle : un GPT écrit à la main, entraîné ici

**2026-10-19 · KAIROS · branche `arena/c9070fe4-scienceai`**
Code : `demo/tinygpt.py` · Corpus : `demo/build_corpus.py`
Sorties : `results/tinygpt-train.txt`, `results/tinygpt-echantillons.txt`,
`figures/tinygpt-courbe.svg` · Poids : `models/tinygpt.npz` (15,7 Mo)

---

## 1. Ce que c'est, sans emballage

Un **vrai transformer**, avec :

- des embeddings de tokens et de positions ;
- **4 couches** d'attention multi-têtes **causale** (4 têtes, dimension 224) ;
- LayerNorm avant chaque sous-bloc, connexions résiduelles ;
- un MLP GELU (224 → 896 → 224) ;
- une tête de sortie **liée** aux embeddings d'entrée ;
- **4 241 216 paramètres** ;
- une **passe arrière écrite à la main** (aucun autograd), vérifiée par
  différences finies : écart relatif **1e-8** ;
- AdamW, cosine schedule, clipping du gradient, évaluation sur un jeu de
  validation tenu à l'écart.

Il tourne avec **numpy seul**. Pas de PyTorch, pas de GPU, pas de
HuggingFace. 2 vCPU, 3 Go de RAM.

Ce n'est **pas** un grand modèle : 4,2 M de paramètres, c'est 30 fois moins
que GPT-2 small (124 M) et 200 fois moins qu'un 1 B. Mais c'est un modèle
réellement entraîné, pas un assemblage de règles.

## 2. Le corpus

`demo/build_corpus.py` télécharge des livres français du domaine public
depuis le miroir GitHub de Project Gutenberg (GITenberg), extrait le texte et
retire l'en-tête Gutenberg. Aucun jeu de données propriétaire, aucune
dépendance à HuggingFace (bloqué dans cette sandbox).

| Métrique | Valeur |
|---|---:|
| Livres retenus | 13 (sur 20 tentatives) |
| Caractères | 10 249 253 (10,2 Mo) |
| Tokens (vocabulaire 8 000) | 2 360 827 |
| Tokens d'entraînement | 2 313 611 |
| Tokens de validation | 47 216 |

Œuvres : *Les Misérables*, *Le Comte de Monte-Cristo*, *Les Trois
Mousquetaires*, *Vingt ans après*, *Notre-Dame de Paris*, *Madame Bovary*,
*L'Assommoir*, *Candide*, *Vingt mille lieues sous les mers*…

Le filtrage est automatique : un livre est rejeté s'il n'est pas en français
(test de fréquence sur les mots-outils), trop court, ou introuvable. C'est
ainsi qu'un tome espagnol a été écarté sans intervention.

## 3. L'entraînement

| Paramètre | Valeur |
|---|---|
| Tokens par pas | 24 × 128 = 3 072 |
| Pas | 1 600 |
| Tokens vus | 4 915 200 (≈ 2 époques) |
| LR max / min | 7e-4 / 3,5e-5, échauffement 80 pas |
| Weight decay | 0,05 (matrices seulement) |
| Débit | ≈ 2 900 tokens/s |
| **Durée** | **28,9 minutes** |

Courbe (`figures/tinygpt-courbe.svg`) :

| Pas | Perte train | Perte validation | Perplexité |
|---:|---:|---:|---:|
| 100 | 6,49 | 5,20 | 180,9 |
| 300 | 4,48 | 4,66 | 105,6 |
| 600 | 4,13 | 4,35 | 77,5 |
| 1000 | 3,89 | 4,15 | 63,4 |
| **1600** | **3,76** | **4,03** | **56,3** |

En bits par caractère : **1,35**. À titre de repère, un GPT-2 small fait
environ 0,7 bpc sur de l'anglais ; 1,35 bpc pour un modèle 30 fois plus
petit, sur du français et avec 2,3 M de tokens, est cohérent.

L'écart entre perte d'entraînement (3,76) et de validation (4,03) montre un
début de surapprentissage : le modèle a 4,2 M de paramètres pour 2,3 M de
tokens. **Il est limité par les données, pas par le calcul.**

## 4. Ce qu'il écrit

Échantillons réels, température 0,75, graine 42
(`results/tinygpt-echantillons.txt`) :

> **« il était une fois »**
> Il était une fois qu'elle avait dit: « ma femme, je vais vous parler de
> vous dire, mon ami, vous pouvez être toujours le temps, et je n'ai pas été
> à vous dire. « — Oui, monsieur, dit la reine, un de mes amis, qui s'en est
> fait un grand monde. — Le, dit

> **« le comte de Monte-Cristo »**
> Le comte de monte-cristo. — Oui, monseigneur, mais que je ne vois pas
> qu'il y a encore les paroles de morrel; mais, je l'ai été passé, je ne vous
> ai pas fait un peu si grand. — Je vous suis, dit, mais je me crois, je vous
> en savez que je suis si c'est-à

> **« elle regarda »**
> Elle regarda sa tête; elle se rappela son front, qu'elle lui tenait un
> moment de la rue de son cheval s'élança à l'heure de son fauteuil, et un
> d'un autre instant, lui faisait un sourire qu'il avait dit. — Oui,
> monsieur, dit-il.

**Ce qui marche** : l'orthographe, la morphologie, la ponctuation, la
typographie du dialogue (tirets, guillemets), les noms propres du corpus
(d'Artagnan, Aramis, Athos, Morrel, Villefort, Valentine), et des groupes de
mots grammaticalement valides sur 5 à 15 mots.

**Ce qui ne marche pas** : la cohérence au-delà d'une phrase, la logique, les
faits. Le modèle enchaîne des propositions plausibles ; il ne suit aucune
intention.

Un détail typographique trahit la mécanique : `monte-cristo` reste en
minuscules, parce que le corpus a été mis en minuscules pour réduire le
vocabulaire. Le modèle n'a jamais vu de majuscules ; c'est la fonction
d'affichage qui recapitalise après un point.

## 5. La comparaison honnête

| | Ce modèle | Tri-système (compétences) | HECATON-I (projeté) |
|---|---:|---:|---:|
| Paramètres | 4,2 M | 0 | 372 B |
| Répond à « 17 × 23 » | parfois, au hasard | **391, vérifié** | probablement |
| Écrit une phrase française | oui | non | oui |
| Coût par réponse | 370 ms · 60 tokens | 0,2 ms | ~8 s |
| Vérifiable | non | oui | non |
| Entraîné ici | **oui, 29 min** | non | non |

Les deux systèmes ne font pas le même travail. Le modèle **génère** ; le
tri-système **répond juste**. Le programme consiste à les combiner : le petit
modèle pour comprendre la demande et composer, les compétences pour exécuter
et garantir.

## 6. Ce que ça change dans le débat

Le 2026-10-19 au matin, ma position était : « un cortex de 8 B, une
bibliothèque de 55 Go ». Ce soir, la démonstration montre deux choses :

1. **Le calcul n'est pas le mur.** 29 minutes sur 2 vCPU suffisent à
   entraîner un transformer qui écrit du français. Le mur, c'est **les
   données** : 2,3 M de tokens pour 4,2 M de paramètres, alors qu'il en
   faudrait 40 fois plus pour saturer le modèle.
2. **Un petit modèle ne peut pas être vérifié.** Il n'y a aucun moyen de
   garantir que « 17 × 23 » donne 391 en interrogeant ce modèle. La seule
   garantie vient du code. C'est l'argument central du tri-système, et il
   sort renforcé de l'expérience.

## 7. Reproduire

```bash
python3 -m venv .venv && .venv/bin/pip install numpy
python3 demo/build_corpus.py --out data/corpus-fr.txt   # 25 s, 13 livres
python3 demo/tinygpt.py gradcheck                        # vérifie la passe arrière
python3 demo/tinygpt.py train                            # 29 min sur 2 vCPU
python3 demo/tinygpt.py sample "il était une fois"
python3 demo/server.py                                   # interface web
```

Le checkpoint est fourni (`models/tinygpt.npz`, 15,7 Mo) : la génération
fonctionne sans réentraîner.

## 8. Ce qui serait la suite, si on continue

| Levier | Gain attendu | Coût |
|---|---|---|
| 10× plus de données (130 livres) | perte 4,0 → ~3,6 | 1 h de téléchargement |
| BPE 16 k au lieu des mots | meilleure couverture, moins d'`<unk>` | 2 h de code |
| 8 couches, d=320 | perte ~3,5 | 4 h d'entraînement |
| Quantification int8 du checkpoint | 15,7 Mo → 4 Mo | 1 h |

Aucun de ces leviers ne change la nature du résultat : à cette échelle, on
obtient un modèle qui **imite la langue**, pas un modèle qui **sait des
choses**. Pour cela il faut des ordres de grandeur de plus en données et en
calcul — et c'est précisément ce que le budget de 100 Go ne donne pas.
