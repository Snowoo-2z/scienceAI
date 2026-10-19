# KAIROS — HECATON-I · résumé en une page (historique)

> **Ceci est la baseline de compression, pas le programme actif.**
> Le programme actif est [`06-PROGRAMME-II.md`](06-PROGRAMME-II.md).


**2026-10-19 · branche `arena/c9070fe4-scienceai` · W1-bis**
Détail : `docs/00` à `docs/05` · chiffres : `results/*.txt` · outil : `tools/hecaton_budget.py`

---

## Le point de désaccord, posé franchement

Le document partagé est construit sur **100 Mo de RSS**. Mon brief parle de
**« une centaine de Go »**. Entre les deux : un facteur 1 000, et deux
physiques différentes.

| | 100 Mo | 100 Go |
|---|---|---|
| paramètres possibles (4,5 bits) | 85 M | 180 B |
| paramètres possibles (1,7 bit) | 200 M | 420 B |
| ce qui limite | le runtime (26 Mo = 26 % du budget) | le noyau de calcul |
| levier | données + distillation | sparsité + quantification |

Les deux sont défendables. **Il faut trancher une fois** (ADR-006, question à
l'humain). En attendant, j'ai couvert la branche 100 Go et j'ai fourni
l'équation qui relie les deux.

## Ce que j'ai construit

- `tools/hecaton_budget.py` — calculateur mémoire / débit / énergie /
  entraînement, stdlib pure, 6 sous-commandes.
- `tools/bwprobe.c` — bande passante et localité (mesuré).
- `tools/tgemv.c` — débit d'un noyau ternaire (mesuré).
- `configs/` — 6 configurations, de 73 M à 372 B.
- `results/` — toutes les sorties brutes, reproductibles.

## Résultat 1 — un 372 B tient dans 85 Go

**HECATON-420A6** : 64 couches, `d=2048`, **896 experts top-10**, attention
hybride (48 DeltaNet + 16 MLA), 8 couches mémoire, experts **ternaires**.

| | valeur |
|---|---:|
| paramètres totaux | **372,3 B** |
| paramètres actifs / token | **6,0 B** (1,6 %) |
| RAM à 32 k | **84,9 Go** |
| octets lus / token | **2,05 Go** |
| débit (noyau 800 GOP/s, Strix Halo) | **49 tok/s** |
| énergie | 2,00 J/token (92 % = la plateforme) |

Variante **HECATON-215A6** : 213 B, **49,5 Go**, **même débit**. +75 % de
capacité pour 0 tok/s de moins — c'est le levier `Λ` en action.

## Résultat 2 — le goulot n'est pas la mémoire, c'est le noyau

À 1,58 bit, la formule `tok/s = BW / octets` est **fausse** : le calcul
domine. Seuil exact :

```
GOPS ≥ 16 · BW / bits
```

À 1,7 bit et 166 Go/s utiles, il faut **1 564 GOP/s** pour être
mémoire-borné. Mesuré ici : **8,5 GOP/s sur 2 vCPU** (noyau C naïf). Un
noyau AVX-512 bien écrit en fait 300 à 700 sur 16 cœurs : **on est encore
2 à 5× en dessous du seuil**.

**Conséquence stratégique : le noyau ternaire (W3) est le chemin critique,
pas l'architecture.** C'est une bonne nouvelle : le calcul progresse avec
les générations de CPU, la bande passante beaucoup moins vite.

## Résultat 3 — deux résultats négatifs qui économisent des semaines

Mesuré avec `bwprobe` (VM 2 vCPU, 1 Go de tas) :

1. **La dispersion des experts ne coûte rien.** Gather de 10 experts
   dispersés dans 512 : **105-107 %** du débit séquentiel. L'« expert
   clustering » est inutile.
2. **Trier les indices n'aide pas** (±2 %). Optimisation à ne pas écrire.
3. **Le vrai danger est sous 4 Ko** : 13 % du débit à 64 o, 56 % à 1 Ko,
   80 % à 4 Ko, 97 % à 16 Ko. Il faut aligner le KV, les embeddings et les
   échelles de quantification, pas les experts.

## Résultat 4 — le mur est l'entraînement, pas l'inférence

`6 · N_actif · D` : 1 T tokens = 25 000 H100-h = **63 k$**. 2 T = 126 k$.

Mais le vrai problème est ailleurs : **tokens par expert**.

| modèle | experts (couches×E) | tokens | tokens/expert |
|---|---:|---:|---:|
| Qwen3-Next-80B | 24 576 | 15 T | **610 M** |
| DeepSeek-V3 | 14 848 | 14,8 T | 997 M |
| **HECATON-420A6** | 57 344 | 2 T | **35 M** |
| HECATON-215A6 | 32 768 | 2 T | 61 M |

**On ne peut pas entraîner 372 B de capacité avec 2 T tokens.** Il faudrait
10 à 20 T tokens (600 k$ à 1,2 M$). La cible réaliste est **215 B**, et même
là on est 10× sous les références.

## Recommandation

**Ne pas construire HECATON d'abord.** Faire E-K8 :

> Quantifier **Qwen3-Next-80B-A3B** en ternaire, réparer par QAT sur 100 Md
> tokens, mesurer la perte réelle. Coût : **5 000 $, 3 semaines.**

Si la perte est < 5 points → le ternaire est validé, HECATON devient un
objectif crédible. Si elle est > 10 points → le régime 100 Go reste à 4 bits,
la capacité tombe à 180 B et HECATON perd son avantage.

## Ce que je demande

| # | Question | À qui |
|---|---|---|
| 1 | 100 Mo ou 100 Go ? | `9c2ae177`, humain |
| 2 | packing ternaire réel (bits effectifs, échelles) | `Silex` |
| 3 | 50-100 Md tokens de réparation : quel corpus ? | W2 |
| 4 | adopter `GOPS ≥ 16·BW/bits` comme test de conception | W5 |
| 5 | mesurer `J/réponse juste` sur les deux régimes | W5/W6 |

## Ce qui me ferait abandonner

E-K2 (ternaire > 5 pts de perte), E-K3 (noyau < 400 GOP/s), E-K8 (> 10 pts
de perte), E-K9 (le 75 M a un meilleur `J/réponse juste`). Les quatre
critères sont écrits à l'avance dans `docs/05-EXPERIENCES-RISQUES.md`.

Même en cas d'échec, la loi de budget, le calculateur et les deux
microbenchmarks restent vrais et réutilisables.
