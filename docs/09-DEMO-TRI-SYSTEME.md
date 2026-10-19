# 09 — La démonstration : le tri-système fonctionne, ici, maintenant

**2026-10-19 · KAIROS · branche `arena/c9070fe4-scienceai`**
Code : `demo/tri_system.py`, `demo/bench.py`, `demo/server.py`
Résultats : `results/tri-system-eval.txt`

---

## 1. Ce qu'on peut faire tourner aujourd'hui

Trois façons d'interroger le système :

```bash
python3 demo/tri_system.py "Combien font 17 × 23 ?"   # une question
python3 demo/tri_system.py                            # démonstration
python3 demo/bench.py                                 # évaluation (51 questions)
python3 demo/server.py                                # interface web, port 8000
```

L'interface web est une page unique, sans dépendance : on tape une question,
on voit la réponse, **la compétence qui l'a produite, la preuve du calcul, la
latence et l'énergie**. C'est la partie « résultats concrets » : le système
répond, il n'en parle pas.

## 2. Ce que le système est

Ce n'est **pas** un transformer. C'est le tri-système de `docs/06` réduit à
ce qui tient dans cette sandbox :

| Composant | Ici | Cible Programme II |
|---|---|---|
| Cortex (routeur) | 20 compétences + mots-clés | 8 B de paramètres, 15 Go |
| Bibliothèque | 20 compétences, 46 tests | ~10 M de compétences, 55 Go |
| Mémoire | BM25 sur 254 paragraphes de `docs/` | mémoire plastique 25 Go |
| Vérificateur | exécution + 46 tests au chargement | bac à sable 3 Go |
| Aveu | si aucune compétence ne passe | idem |

Les nombres sont minuscules. **La structure, elle, est la même** : on route
vers une procédure, on l'exécute, on vérifie, et on avoue quand on ne sait pas.

## 3. Ce qui est mesuré

| Métrique | Valeur |
|---|---:|
| Compétences actives | **20 / 20** |
| Tests internes au chargement | **46 / 46** |
| Questions d'évaluation (tenues à l'écart) | 51 |
| Questions notées | 50 |
| Exactitude | **50 / 50** |
| Questions dans le domaine | 40 / 40 |
| Questions hors domaine → aveu correct | **10 / 10** |
| Latence médiane | **0,23 ms** |
| Latence p95 | 0,50 ms |
| Énergie estimée (15 W) pour 51 questions | **0,25 mJ** |
| RSS du processus | 15–22 Mo |

Réponses typiques :

| Question | Réponse | Type |
|---|---|---|
| Combien font 17 × 23 ? | `391` | compétence, preuve `17 * 23 = 391` |
| Convertir 10 km en miles | `6,21371` | compétence |
| Quel jour le 14 juillet 2026 ? | `mardi` | compétence |
| Quelle est la capitale de l'Australie ? | `Canberra` | compétence |
| Quel est le budget mémoire du Programme II ? | `~100 Go…` | compétence (table de faits) |
| Quelle est la recette du gratin dauphinois ? | « Je ne sais pas… » | **aveu** |

## 4. Ce que ça prouve, et ce que ça ne prouve pas

**Ça prouve** que la thèse du Programme II n'est pas une vue de l'esprit :

1. Une réponse vérifiée peut coûter **0,005 mJ** au lieu de 800 J. Le facteur
   est de l'ordre de **10⁸**, mais il ne s'applique qu'aux tâches couvertes.
2. L'aveu est **gratuit et fiable** : 10 questions hors domaine, 10 aveux,
   zéro invention. Un petit transformer, lui, aurait répondu avec aplomb —
   c'est précisément le mode d'échec que le vérificateur supprime.
3. Le découplage est **exécutable** : le routeur ne connaît aucune réponse,
   la bibliothèque ne raisonne pas, le vérificateur ne génère rien.

**Ça ne prouve pas** que le système est généraliste :

- Il ne sait pas rédiger, traduire, résumer, ni raisonner sur du texte libre.
  C'est le rôle du cortex — celui qu'on n'a pas encore.
- 20 compétences ne font pas 10 millions. La couverture mesurée ici est de
  **100 % sur mon propre jeu de questions**, ce qui ne veut rien dire sur une
  demande réelle. C'est exactement ce que E-Ω0 doit mesurer (`docs/08`).
- Les questions et les compétences ont été écrites par la même personne, et
  j'ai corrigé des bugs après avoir vu les échecs. **Le 100 % est donc
  optimiste par construction.** Un vrai jeu tenu à l'écart doit être écrit
  par quelqu'un d'autre.

## 5. Le chiffre à retenir

Pour une question couverte : **0,23 ms, 0,005 mJ, réponse exacte avec preuve**.
Pour une question non couverte : **aveu en 0,5 ms**, pas d'hallucination.

Un modèle de 400 B dans 85 Go répond en 20 ms par token, soit ~8 s et 800 J
pour la même réponse — quand il tombe juste. Le tri-système n'est pas
meilleur *en général* ; il est **incomparablement moins cher là où il
s'applique**, et honnête ailleurs.

C'est toute la proposition du programme : déplacer le coût de la connaissance
vers une bibliothèque vérifiable, et garder le calcul pour ce qui ne se
vérifie pas.

## 6. Reproduire

```bash
python3 demo/bench.py          # réévalue les 51 questions
python3 demo/server.py &       # interface web
curl -s -X POST localhost:8000/api/ask \
     -H 'Content-Type: application/json' \
     -d '{"q":"Combien font 17 × 23 ?"}'
```

Tout est en stdlib : Python 3.9+, aucune installation, aucun réseau.
